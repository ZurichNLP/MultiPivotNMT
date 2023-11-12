import json
import logging
import os
import warnings

from sqlitedict import SqliteDict
from pathlib import Path
from typing import List, Union, Tuple, Set, Optional

from tqdm import tqdm


class TranslationModel:

    def __str__(self):
        raise NotImplementedError

    def translate(self,
                  tgt_lang: str,
                  source_sentences: Union[str, List[str]],
                  src_lang: str = None,
                  return_score: bool = False,
                  batch_size: int = 8,
                  use_cache: bool = True,
                  num_beams: int = 5,
                  **kwargs,
                  ) -> Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]:
        """
        :param tgt_lang: Language code of the target language
        :param source_sentences: A sentence or list of sentences
        :param src_lang: Language code of the source language (not needed for some multilingual models)
        :param return score: If true, return a tuple where the second element is sequence-level score of the translation
        :param batch_size
        :param use_cache
        :param kwargs
        :return: A sentence or list of sentences
        """
        if isinstance(source_sentences, str):
            source_sentences_list = [source_sentences]
        elif isinstance(source_sentences, list):
            source_sentences_list = source_sentences
        else:
            raise ValueError

        if use_cache:
            if kwargs or num_beams != 5:
                raise NotImplementedError
            cached_translations_list = []
            with self.load_cache() as cache:
                for source_sentence in source_sentences_list:
                    translation = cache.get(f"{(src_lang + '_') if src_lang is not None else ''}{tgt_lang}_"
                                            f"translate{'_score' if return_score else ''}_{source_sentence}", None)
                    cached_translations_list.append(translation)
            full_source_sentences_list = source_sentences_list
            source_sentences_list = [
                source_sentence for source_sentence, cached_translation
                in zip(full_source_sentences_list, cached_translations_list)
                if cached_translation is None
            ]

        self._set_tgt_lang(tgt_lang)
        if self.requires_src_lang:
            if src_lang is None:
                warnings.warn(f"NMT model {self} requires the src language. Assuming 'en'; override with `src_lang`")
                src_lang = "en"
            self._set_src_lang(src_lang)
        translations_list = self._translate(source_sentences_list, return_score, batch_size, num_beams=num_beams, **kwargs)
        assert len(translations_list) == len(source_sentences_list)

        if use_cache:
            cache_update = dict()
            for i, cached_translation in enumerate(cached_translations_list):
                if cached_translation is not None:
                    translations_list.insert(i, cached_translation)
                else:
                    cache_update[f"{(src_lang + '_') if src_lang is not None else ''}{tgt_lang}_" \
                                 f"translate{'_score' if return_score else ''}_" \
                                 f"{full_source_sentences_list[i]}"] = translations_list[i]
            if cache_update:
                with self.load_cache() as cache:
                    cache.update(cache_update)
                    cache.commit()

        if isinstance(source_sentences, str):
            translations = translations_list[0]
        else:
            translations = translations_list
        return translations

    def translate_pivot(self,
                        tgt_lang: str,
                        source_sentences: Union[str, List[str]],
                        pivots: str = "en",
                        src_lang: str = None,
                        batch_size: int = 8,
                        use_cache: bool = True,
                        simple_avg=False,
                        **kwargs,
                        ) -> Union[str, List[str]]:
        pivots = pivots[0]
        pivot_translations = self.translate(
            pivots,
            source_sentences,
            src_lang,
            return_score=False,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs,
        )
        return self.translate(
            tgt_lang,
            pivot_translations,
            pivots,
            return_score=False,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs,
        )

    def translate_bridge(self,
                        tgt_lang: str,
                        source_sentences: Union[str, List[str]],
                        pivots: str = "en",
                        src_lang: str = None,
                        batch_size: int = 8,
                        use_cache: bool = True,
                        **kwargs,
                        ) -> Union[str, List[str]]:
        bridge_source = pivots[0]
        bridge_target = pivots[1]

        bridge_translations = self.translate(
            bridge_source,
            source_sentences,
            src_lang,
            return_score=False,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs,
        )

        bridge_t_translations = self.translate(
            bridge_target,
            bridge_translations,
            bridge_source,
            return_score=False,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs,
        )
        
        return self.translate(
            tgt_lang,
            bridge_t_translations,
            bridge_target,
            return_score=False,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs,
        )

    def translate_multi_pivot(self,
                              tgt_lang: str,
                              source_sentences: Union[str, List[str]],
                              src_lang: str,
                              pivots: List[str],
                              simple_avg: bool = False,
                              batch_size: int = 16,
                              use_cache: bool = False,
                              **kwargs,
                              ) -> Union[str, List[str]]:
        pivot_languages = pivots
        num_pivot = len(pivot_languages)
        all_pivot_translations = []
        for pivot_lang in pivot_languages:
            pivot_translations = self.translate(
                pivot_lang,
                [source_sentences] if isinstance(source_sentences, str) else source_sentences,
                src_lang,
                return_score=False,
                batch_size=batch_size,
                use_cache=use_cache,
                **kwargs,
            )
            all_pivot_translations.append(pivot_translations)

        translations = []
        for i, segment_pivot_translations in enumerate(tqdm(list(zip(*all_pivot_translations)))):
            segment_pivot_translations = list(segment_pivot_translations)
            assert len(segment_pivot_translations) == num_pivot
            translation = self.translate_multi_source(
                tgt_lang=tgt_lang,
                multi_source_sentences=segment_pivot_translations,
                src_langs=pivot_languages,
                use_cache=use_cache,
                simple_avg=simple_avg,
                **kwargs,
            )
            translations.append(translation)
        if isinstance(source_sentences, str):
            return translations[0]
        else:
            return translations

    def translate_iterative(self,
                            tgt_lang: str,
                            source_sentences: Union[str, List[str]],
                            src_lang: str,
                            num_iterations: int = 10,
                            batch_size: int = 8,
                            use_cache: bool = True,
                            num_beams_initial: int = 5,
                            num_beams_repeat: int = 5,
                            **kwargs,
                            ) -> Union[str, List[str]]:
        translations = self.translate(
            tgt_lang=tgt_lang,
            source_sentences=[source_sentences] if isinstance(source_sentences, str) else source_sentences,
            src_lang=src_lang,
            batch_size=batch_size,
            use_cache=use_cache,
            num_beams=num_beams_initial,
            **kwargs,
        )
        logging.info(translations)
        for i in tqdm(list(range(len(translations)))):
            for j in range(1, num_iterations):
                source_sentence = source_sentences if isinstance(source_sentences, str) else source_sentences[i]
                next_translation = self.translate_multi_source(
                    tgt_lang=tgt_lang,
                    multi_source_sentences=[source_sentence, translations[i]],
                    src_langs=[src_lang, tgt_lang],
                    use_cache=use_cache,
                    num_beams=num_beams_repeat,
                    **kwargs,
                )
                logging.info(next_translation)
                if translations[i] == next_translation:  # Unchanged
                    break
                translations[i] = next_translation
        if isinstance(source_sentences, str):
            return translations[0]
        else:
            return translations

    @property
    def supported_languages(self) -> Set[str]:
        raise NotImplementedError

    @property
    def ranked_languages(self) -> List[str]:
        """
        List or sublist of languages ranked by pivot utility in descending order
        """
        raise NotImplementedError

    @property
    def requires_src_lang(self) -> bool:
        """
        Boolean indicating whether the model requires the source language to be specified
        """
        raise NotImplementedError

    def _set_src_lang(self, src_lang: str):
        raise NotImplementedError

    def _set_tgt_lang(self, tgt_lang: str):
        raise NotImplementedError

    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 8,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        raise NotImplementedError

    def _translate_multi_source(self,
                                multi_source_sentences: List[str],
                                src_langs: List[str],
                                tgt_lang: str,
                                src_weights: Optional[List[float]] = None,
                                **kwargs,
                                ) -> str:
        raise NotImplementedError

    def translate_multi_source(self,
                               tgt_lang: str,
                               multi_source_sentences: List[str],
                               src_langs: Optional[List[str]] = None,
                               src_weights: Optional[List[float]] = None,
                               use_cache: bool = False,
                               num_beams: int = 5,
                               **kwargs,
                               ) -> str:
        translation = None
        if use_cache:
            if src_weights or kwargs:
                raise NotImplementedError
            with self.load_cache() as cache:
                translation = cache.get(f"{(','.join(src_langs) + '_') if src_langs is not None else ''}{tgt_lang}_"
                                        f"translate_multi_source_{','.join(multi_source_sentences)}_num_beams_{num_beams}", None)
                cached_translation = translation

        if translation is None:
            self._set_tgt_lang(tgt_lang)
            if self.requires_src_lang:
                assert src_langs is not None
            translation = self._translate_multi_source(multi_source_sentences, src_langs, tgt_lang, src_weights=src_weights, num_beams=num_beams, **kwargs)

        if use_cache:
            cache_update = dict()
            if cached_translation is None:
                cache_update[f"{(','.join(src_langs) + '_') if src_langs is not None else ''}{tgt_lang}_"
                                f"translate_multi_source_{','.join(multi_source_sentences)}_num_beams_{num_beams}"] = translation
                with self.load_cache() as cache:
                    cache.update(cache_update)
                    cache.commit()

        return translation

    @property
    def cache_path(self) -> Path:
        """
        :return: Path of the SQLite database where the translations and scores are cached
        """
        cache_dir = Path(os.getenv("MULTI_PIVOT_CACHE", Path.home() / "multi_pivot"))
        if not cache_dir.exists():
            os.mkdir(cache_dir)
        return cache_dir / (str(self).replace("/", "_") + ".sqlite")

    def load_cache(self) -> SqliteDict:
        """
        :return: A connection to the SQLite database where the translations and scores are cached
        """
        return SqliteDict(self.cache_path, timeout=15, encode=json.dumps, decode=json.loads)


def load_translation_model(name: str, **kwargs) -> TranslationModel:
    """
    Convenience function to load a :class: TranslationModel using a shorthand name of the model
    """
    if name == "m2m100_418M":
        from translation_models.m2m100 import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_418M", **kwargs)
    elif name == "m2m100_1.2B":
        from translation_models.m2m100 import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_1.2B", **kwargs)
    elif name == "small100":
        from translation_models.small100 import SMaLL100Model
        translation_model = SMaLL100Model(model_name_or_path="alirezamsh/small100", **kwargs)
    else:
        raise NotImplementedError
    return translation_model
