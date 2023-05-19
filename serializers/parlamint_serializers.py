import csv
import json
import os.path
from dataclasses import field
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
from pydantic import parse_obj_as, Field
from pydantic.dataclasses import dataclass
from enum import Enum
from pydantic.json import pydantic_encoder


class Config:
    arbitrary_types_allowed = True


class ParlaMintLanguage(str, Enum):
    ES = 'ES-GA'
    GB = 'GB'
    HU = 'HU'
    SI = 'SI'
    UA = 'UA'

    def get_iso_code(self) -> str:
        return {
            ParlaMintLanguage.ES: 'es',
            ParlaMintLanguage.GB: 'en',
            ParlaMintLanguage.HU: 'hu',
            ParlaMintLanguage.SI: 'sl',
            ParlaMintLanguage.UA: 'uk',
        }[self]


@dataclass(config=Config)
class ParlaMintMetadata:
    title: Optional[str]
    date: Optional[str]
    body: Optional[str]
    term: Optional[str]
    session: Optional[str]
    meeting: Optional[str]
    sitting: Optional[str]
    agenda: Optional[str]
    subcorpus: Optional[str]
    speaker_role: Optional[str]
    speaker_mp: Optional[str]
    speaker_minister: Optional[str]
    speaker_party: Optional[str]
    speaker_party_name: Optional[str]
    party_status: Optional[str]
    speaker_name: Optional[str]
    speaker_gender: Optional[str]
    speaker_birth: Optional[str]


@dataclass(config=Config)
class ParlaMintDataset:
    @dataclass(config=Config)
    class Speech:
        @dataclass(config=Config)
        class Sentence:
            text: str
            interjection: bool = False
            interjection_speaker: Optional[str] = None
            embedding: Optional[np.ndarray] = field(default=None, metadata=dict(exclude=True))

        id: str
        text: str
        sentences: Optional[list[Sentence]] = None
        embeddings: Optional[np.ndarray] = field(default=None, metadata=dict(exclude=True))
        artefacts_base: Optional[Path] = field(default=None, metadata=dict(exclude=True))

        @property
        def artefact_embeddings(self) -> Path:
            return self.artefacts_base / f'{self.id}.npy'

        def load_embeddings(self):
            if self.artefact_embeddings.exists():
                with open(self.artefact_embeddings, 'rb') as f:
                    self.embeddings = np.load(f)
                    for i, sentence in enumerate(self.sentences):
                        sentence.embedding = self.embeddings[i]
            else:
                raise FileNotFoundError(f'Embeddings for speech {self.id} not found')

        def save_embeddings(self):
            if not self.embeddings:
                raise ValueError('No embeddings to save')
            self.artefacts_base.mkdir(parents=True, exist_ok=True)
            with open(self.artefact_embeddings, 'wb') as f:
                np.save(f, self.embeddings)

    path: str
    date: str
    language: ParlaMintLanguage
    metadata_path: str

    @property
    def year(self) -> int:
        return int(self.date[:4])

    def get_metadata(self) -> dict[str, ParlaMintMetadata]:
        metadata = {}
        with open(self.metadata_path) as f:
            metadata_file = csv.reader(f, delimiter='\t')
            header = [c.lower() for c in next(metadata_file)[1:]]
            for row in metadata_file:
                metadata[row[0]] = ParlaMintMetadata(**dict(zip(header, row[1:])))
        return metadata

    def speeches(self, ignore_artefact=False, load_embeddings=True) -> Iterable[Speech]:
        if not ignore_artefact and self.artefact_jsonl.exists():
            with open(self.artefact_jsonl) as f:
                artefacts_base = self.artefacts_base
                for i, line in enumerate(f):
                    speech = parse_obj_as(self.Speech, json.loads(line))
                    speech.artefacts_base = artefacts_base
                    if load_embeddings:
                        try:
                            speech.load_embeddings()
                        except FileNotFoundError:
                            print(f'Failed to load embeddings for speech {speech.id}')
                    yield speech
        with open(self.path) as f:
            for row in csv.reader(f, delimiter='\t'):
                yield self.Speech(id=row[0], text=row[1])

    @property
    def artefacts_base(self) -> Path:
        return Path('artefacts') / self.language.value / str(self.year) / os.path.splitext(Path(self.path).name)[0]

    @property
    def artefact_jsonl(self) -> Path:
        return self.artefacts_base.with_suffix('.jsonl')

    def save_speeches(self, speeches: list[Speech]):
        self.artefact_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.artefact_jsonl, 'w') as f:
            for speech in speeches:
                f.write(json.dumps(speech, default=pydantic_encoder) + '\n')


@dataclass(config=Config)
class ParlaMint:
    artefact = 'artefacts/ParlaMint.json'
    languages: dict[ParlaMintLanguage, list[ParlaMintDataset]] = field(default_factory=dict)

    def save(self):
        with open(self.artefact, 'w') as f:
            json.dump(self, f, default=pydantic_encoder)

    @classmethod
    def load(cls) -> 'ParlaMint':
        return parse_obj_as(cls, json.load(open(cls.artefact)))
