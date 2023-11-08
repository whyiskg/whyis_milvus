from whyis.plugin import Plugin, NanopublicationListener
from whyis.database import driver
from whyis.namespace import NS
from pymilvus import connections, db, Collection, utility
from flask import current_app
import rdflib
import json
import collections
from functools import reduce
import numpy as np

from pymilvus import CollectionSchema, FieldSchema, DataType

whyis = NS.whyis

class VectorDBException(Exception):
    '''Raise an exception when something goes wrong in the vector database.'''

class VectorSpace:

    _collection = None

    field_names = ["subject", "graph", "extent", "v"]

    def __init__(self, vs_resource, db):
        self.resource = vs_resource
        self.identifier = vs_resource.identifier
        self.db = db
        self.collection_id = self.resource.value(NS.dc.identifier).value
        self.extent = self.resource.value(NS.whyis.hasExtent)
        self.extent = tuple(json.loads(str(self.extent)))
        self.dimensions = reduce(lambda a, b: a*b, self.extent)
        print(self.extent, self.dimensions)

        self.distance_metric = self.resource.value(whyis.hasDistanceMetric)
        if self.distance_metric is not None:
            self.distance_metric = self.distance_metric.value
        self.index_type = self.resource.value(whyis.hasIndexType)
        if self.index_type is not None:
            self.index_type = str(self.index_type)

        self.field_names.extend(list(self.resource[whyis.hasDynamicField]))

    @property
    def collection(self):
        if self._collection is None:
            if utility.has_collection(self.collection_id, using=self.db.name):
               self._collection = Collection(self.collection_id, using=self.db.name)
               print("Loading %s" % self.identifier)
               #self._collection.load()
               print("Done Loading %s" % self.identifier)
               print("%s has %s vectors." % (self.identifier, self._collection.num_entities))
            else:
                extent_dimensions = len(self.extent)
                dimensions = self.dimensions
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id = True
                    ),
                    FieldSchema(name="subject",
                                dtype=DataType.VARCHAR,
                                max_length=65535),
                    FieldSchema(name="graph",
                                dtype=DataType.VARCHAR,
                                max_length=65535),
                    FieldSchema(
                        name="v",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=dimensions
                    ),
                    # FieldSchema(
                    #     name="extent",
                    #     dtype=DataType.FLOAT_VECTOR,
                    #     dim=extent_dimensions
                    # )
                ]
                print(fields)
                schema = CollectionSchema(
                    fields=fields,
                    primary_field='id',
                    enable_dynamic_field=True
                )
                coll_name = self.collection_id
                using = self.db.name
                self._collection = Collection(
                    name=coll_name,
                    schema=schema,
                    using=using
                )
                metric_type = self.distance_metric
                index_type = self.index_type
                self._collection.create_index(
                    field_name="v",
                    index_params={
                        "metric_type" : metric_type,
                        "index_type" : index_type
                    }
                )
                self._collection.create_index(field_name="subject")
                self._collection.create_index(field_name="graph")
                self._collection.load()
        return self._collection

    def prepare_result(self, result):
        result['tensor'] = self.db.to_tensor(result['v'], self.extent).tolist()
        del result['v']
        result['space'] = self.identifier

        return result

    def similar(self, subject, graph=None, limit=10, offset=0):
        matches = self.get(subject, graph)
        results = []
        for match in matches:
            tensor = match['tensor']
            results.extend(self.search(tensor, limit=limit, offset=offset))
        return results

    def get(self, subject=None, graph=None):
        params = dict(subject=subject, graph=graph)
        exp = ' and '.join([
            '%s == "%s"'%(key, value)
            for key, value in params.items()
            if value is not None])
        print(exp)
        results = self.collection.query(
            expr=exp,
            output_fields=self.field_names,
            consistency_level="Eventually"
        )
        results = [self.prepare_result(r) for r in results]

        return results

    def search(self, tensor, limit, offset=0):
        search_params = {
            "metric_type": self.distance_metric,
            "offset": offset,
        }
        vector, extent = self.db.to_vector(tensor)

        print("searching %s"%self.identifier)
        result = self.collection.search(
            data=np.array([vector]),
            anns_field="v",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=search_params,
            limit=limit,
            output_fields=self.field_names,
            consistency_level="Eventually"
        )
        print("preparing results for %s"%self.identifier)
        i = 0
        results = []
        for hits in result:
            for hit in hits:
                entity = dict([
                    (field,hit.entity.get(field))
                    for field in self.field_names
                ])

                # entities = self.get(hit.entity.get('subject'),hit.entity.get('graph'))
                # for entity in entities: # should just be one, I think?
                entity['distance'] = hit.score
                entity['rank'] = i + offset
                i += 1
                results.append(entity)

        print("finished searching %s"%self.identifier)
        return results

@driver('milvus')
class MilvusDatabase(NanopublicationListener):

    formats = {
        'json' : NS.rdf.JSON
    }

    vector_parsers = {
        NS.mediaTypes['application/json'] : lambda d: d,
        NS.rdf.JSON : lambda d: d,
    }

    vector_serializers = {
        NS.mediaTypes['application/json'] :
            lambda a: a,
        NS.rdf.JSON :
            lambda a: a,
    }

    _spaces = None

    def __init__(self, config):
        self.db_name = config.get('_database', 'whyis')
        self.user = config.get('_username', None)
        self.password = config.get('_password', None)
        self.name = config.get('_name', "milvus")
        self.host = config.get('_hostname','localhost')
        self.port = config.get('_port', '19530')

        self.default_connection = connections.connect(
            alias=self.name,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        print(db.list_database(using=self.name))
        if self.db_name not in db.list_database(using=self.name):
            db.create_database(self.db_name, using=self.name)
            db.using_database(self.db_name, using=self.name)

    @property
    def spaces(self):
        if self._spaces is None:
            self._spaces = {}
            for uri in current_app.vocab.subjects(rdflib.RDF.type, NS.whyis.VectorSpace):
                print("Adding vector space",uri)
                resource = current_app.vocab.resource(uri)
                self._spaces[uri] = VectorSpace(resource, self)
        return self._spaces

    def _delete_graph(self, uri, space):
        print('deleting')
        deleted = 0
        query_exp = 'graph == "%s"'%uri
        print(query_exp)
        iter = space.collection.query_iterator(
            exp=query_exp,
            limit=10,
            output_fields=['id']
        )
        print("iterating")
        while True:
            print("getting batch")
            results = [x['id'] for x in iter.next()]

            print(results)
            deleted += len(results)
            if len(results) == 0:
                break
            exp = 'id in %s'%json.dumps(results)
            space.collection.delete(exp)
            print("deleted")
        print("done deleting %s vectors" % deleted)
        if len(results) > 0:
            space.collection.flush()
            print("flushed")

    def on_publish(self, nanopub):
        g = rdflib.ConjunctiveGraph(store=nanopub.store)
        assertion_uri = str(nanopub.assertion.identifier)
        for space in self.spaces.values():
            inserts = []
            removes = []
            for s, p, o in nanopub.assertion.triples((None,space.identifier,None)):
                data = self.parse_vector(o)
                tensor = data['tensor']
                vector, extent = self.to_vector(tensor)
                data['v'] = vector
                data['extent'] = extent
                if space.extent is not None and extent != space.extent:
                    print(vector)
                    raise VectorDBException(
                        "Tensor shape (%s) does not match schema tensor shape (%s)." % (extent, space.extent)
                    )
                del data['tensor']
                data['subject'] = str(s)
                data['graph'] = str(nanopub.assertion.identifier)
                inserts.append(data)
                removes.append((s, p, o))
            #self._delete_graph(assertion_uri, space)
            if len(inserts) > 0:
                print("inserting")
                result = space.collection.insert(inserts)
                print("inserted %s entities"% result.insert_count)
                print("primary keys are %s" % result.primary_keys)
                for s, p, o in removes:
                    nanopub.assertion.remove((s, p, o))
                    nanopub.assertion.add((s, NS.whyis.inVectorSpace, space.identifier))
            space.collection.flush()
            print("flushed")
            print("Collection %s now has %s entities."%(space.identifier, space.collection.num_entities))

    def on_retire(self, nanopub):
        assertion_uri = str(nanopub.assertion.identifier)
        spaces = [self.spaces[uri] for uri
                  in set(nanopub.assertion.objects(None, NS.whyis.inVectorSpace))
                  if uri in self.spaces]
        for space in spaces:
            self._delete_graph(assertion_uri, space)
            space.collection.flush()

    def search(self, space, vector, limit, offset=0):
        try:
            s = self.spaces[rdflib.URIRef(space)]
            return s.search(vector, limit, offset=offset)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

    def get(self, space=None, subject=None, graph=None):
        spaces = [space] if space is not None or len(space) == 0 else self.spaces.keys()
        try:
            spaces = [self.spaces[rdflib.URIRef(s)] for s in spaces]
            print(spaces)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

        results = []
        for s in spaces:
            results.extend(s.get(subject, graph))
        return results

    def similar(self, space=None, subject=None, graph=None, limit=10, offset=0):
        spaces = [space] if space is not None or len(space) == 0 else self.spaces.keys()
        try:
            spaces = [self.spaces[rdflib.URIRef(s)] for s in spaces]
            print(spaces)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

        results = []
        for s in spaces:
            results.extend(s.similar(subject, graph, limit, offset))
        return results


    def get_vectors(self, space=None, uri=None, graph=None):
        results = []
        for space in self.spaces.values():
            results.extend(space.get(uri, graph))
        return result

    def parse_vector(self, node):
        data = self.vector_parsers[node.datatype](node.value)
        print(type(data))
        if not isinstance(data, dict):
            data = dict(tensor=data)
        return data

    def serialize_vector(self, tensor, format):
        if format not in self.vector_serializers:
            format = self._formats[format]
        serializer = self.vector_serializers[format]
        return Literal(serializer(tensor), datatype=format)

    def to_vector(self, tensor):
        tensor = np.array(tensor)
        extent = tensor.shape
        vector = tensor.flatten()
        return vector, extent

    def to_tensor(self, vector, extent):
        return np.array(vector).reshape(extent)

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.disconnect(self.name)

class MilvusPlugin(Plugin):

    def create_blueprint(self):
        return None

    def init(self):
        rdflib.term.bind(
            datatype=rdflib.RDF.JSON,
            pythontype=list,
            constructor=json.loads,
            lexicalizer=json.dumps,
            datatype_specific=True
        )
        rdflib.term.bind(
            datatype=NS.mediaTypes['application/json'],
            pythontype=list,
            constructor=json.loads,
            lexicalizer=json.dumps,
            datatype_specific=True
        )
        #driver(MilvusDatabase, "milvus")
