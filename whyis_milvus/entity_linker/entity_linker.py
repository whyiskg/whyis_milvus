from whyis.autonomic import UpdateChangeService
from whyis.namespace import NS
from rdflib.paths import *

from whyis.plugin import Plugin, EntityResolverListener
import rdflib
from flask import current_app

from functools import reduce

from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

import json

prefixes = dict(
    skos = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#"),
    foaf = rdflib.URIRef("http://xmlns.com/foaf/0.1/"),
    text = rdflib.URIRef("http://jena.apache.org/fulltext#"),
    schema = rdflib.URIRef("http://schema.org/"),
    owl = rdflib.OWL,
    rdfs = rdflib.RDFS,
    rdf = rdflib.RDF,
    dc = rdflib.URIRef("http://purl.org/dc/terms/")
)

class MilvusEntityResolverPlugin(Plugin):

    def init(self):
        resolver_db = self.app.config.get('RESOLVER_DB', "milvus")
        resolver_space = self.app.config.get('RESOLVER_SPACE', NS.whyis['vspace/sbert_label'])
        resolver_space = rdflib.URIRef(resolver_space)
        encoder = self.app.config.get('RESOLVER_ENCODER', 'all-mpnet-base-v2')
        resolver = MilvusEntityResolver(resolver_db, resolver_space, encoder)
        self.app.add_listener(resolver)

class MilvusEntityResolver(EntityResolverListener):

    _model = None

    def __init__(self, database="milvus", space=NS.whyis['vspace/sbert_label'], encoder='all-mpnet-base-v2'):
        self._database = database
        self._space = space
        self._encoder_model = encoder

    @property
    def model(self):
        if self._model is None:
            self._model  = SentenceTransformer(self._encoder_model)
        return self._model

    def get_embedding(self, text):
        return self.model.encode(text)

    @property
    def vector_space(self):
        return current_app.databases[self._database].spaces[self._space]

    def filter_type(self, entity, type):
        return (
            rdflib.URIRef(entity),
            NS.rdfs.subClassOf*ZeroOrMore/NS.rdf.type,
            type
        ) in current_app.db

    def types(self, entity):
        return list(
            current_app.db.objects(
                rdflib.URIRef(entity),
                NS.rdfs.subClassOf*ZeroOrMore/NS.rdf.type
            )
        )

    def on_resolve(self, term, type=None, context=None, label=True):
        term_embedding = self.model.encode(term)
        candidates = self.vector_space.search(term_embedding, 10)
        perfect_matches = [c for c in candidates if c['distance'] == 0]


        if type is not None:
            type = rdflib.URIRef(type)
            for c in candidates:
                c['types'] = self.types(c['subject'])
            candidates = [c for c in candidates if type in c['types'] ]

        if len(perfect_matches) > 0 and context is not None:
            context_embedding = self.model.encode(context)
            for c in perfect_matches:
                if 'context_embedding' in c:
                    c['distance'] = cosine(context_embedding, c['context_embedding'])
            perfect_matches = sorted(perfect_matches, key=lambda c: c['distance'])
            candidates = perfect_matches

        for c in candidates:
            if 'types' in c:
                c['types'] = [{'uri':x} for x in c['types']]
            c['node'] = c['subject']
            del c['rank']
            del c['subject']
            if label:
                current_app.labelize(c,'node','preflabel')
                if 'types' in c:
                    c['types'] = [
                        current_app.labelize(x,'uri','label')
                        for x in c['types']
                    ]
        return candidates


class EntityIndexer(UpdateChangeService):
    activity_class = NS.whyis.EntityIndexing # search engine indexing

    predicates = [
        NS.dc.title,
        NS.rdfs.label,
        NS.skos.prefLabel,
        NS.skos.altLabel,
        NS.foaf.name,
        NS.dc.identifier,
        NS.schema.name,
        NS.skos.notation
    ]

    context_datatypes = set([
        None,
        NS.xsd.string,
        NS.xsd.normalizedString,
        NS.rdf.HTML
    ])

    _resolver = None

    @property
    def label_path(self):
        return reduce(lambda x, y: x | y, self.predicates)

    @property
    def resolver(self):
        if self._resolver is None:
            print(self.app.listeners['on_resolve'])
            for resolver in self.app.listeners['on_resolve']:
                if isinstance(resolver, MilvusEntityResolver):
                    self._resolver = resolver
        return self._resolver

    def get_query(self):
        return '''select distinct ?resource where {
            ?resource %s ?literal.
        }''' % self.label_path.n3()

    def get_context(self, i):
        context = []
        for s, p, o in i.graph.triples((i.identifier,None,None)):
            if p not in self.predicates:
                if isinstance(p, rdflib.Literal):
                    if o.datatype in self.context_datatypes:
                        context.append(o.value)
        return '\n'.join(context)

    def process(self, i, o):
        for property in self.predicates:
            for value in i[property]:
                embedding = self.resolver.get_embedding(value.value).tolist()
                r = {
                    "predicate" : str(property),
                    "label" : value.value,
                    "tensor" : embedding
                }
                context = self.get_context(i)
                if len(context) > 0:
                    context_embedding = self.resolver.get_embedding(context).tolist()
                    r['context_embedding'] = context_embedding
                o.add(
                    self.resolver.vector_space.identifier,
                    rdflib.Literal(r, datatype=NS.rdf.JSON)
                )
