"""MIMIR final-caption stack: WHAT, WHEN and HOW are owned by different components.

    WHAT was said    -> lexical evidence: Qwen Omni primary ear (pass A, least biased) and
                        precision ear (pass B, verified spellings as reference only);
                        a bounded model-diverse OpenAI ear only when genuinely needed
    DISPUTED WHAT    -> deterministic agreement, else the Astra caption judge
                        (ai.editor.caption_judge) over the smallest local evidence package
    FROZEN WHAT      -> ``lexical.FrozenTranscript``: immutable tokens + signature; nothing
                        downstream rewrites a word (only the evidence-gated verified-name
                        spelling lock in caption truth, with recorded provenance)
    WHEN it was said -> ONE word-alignment provider per successful run
                        (Qwen3-ForcedAligner by default) over the frozen tokens and the
                        final-short analysis audio; validated, locally re-aligned when a
                        small region fails, otherwise the configured fallback provider
                        re-times the whole transcript (never a mix of two clocks)
    WHICH VOICE      -> acoustic diarization turns, a caption colour per voice, never a
                        person (ai.editor.speaker_caption_support)
    HOW it is shown  -> the deterministic presentation (captions.py / pro_edit)

Modules (nothing heavy is imported by this package itself):

    config           centralized env settings (providers, models, device, dtype)
    audio            the canonical 16 kHz mono PCM analysis audio of the final short
    qwen_omni        DashScope OpenAI-compatible client + JSON evidence contract
    openai_ears      bounded OpenAI transcription (fallback / model-diverse evidence)
    lexical          listening passes, comparison, disputes, judge, freeze
    alignment        provider boundary, validation, local recovery, chunk merge
    legacy_whisper   migration fallback timing provider (isolated Whisper clock)
    final_captions   orchestration for the selected short

Migration map of the pre-Qwen final-caption timing/wording code:

    REMOVED              gpt-transcribe primary + gpt-4o-transcribe cross-check + multi-ear
                         micro votes (speaker_caption_support._transcribe_edited_words,
                         _micro_refine_caption, logprob suspicion, vod_processor caption
                         ears); the ears' text mapped onto Whisper anchors as the normal
                         clock; the micro-vote -> word mapping in caption truth / name lock
    MIGRATION FALLBACK   legacy_whisper: align_text_to_fixed_whisper_clock (exact anchors,
                         local insert/replace, tail borrowing) + the exact-PCM phrase-start
                         guard, behind the alignment-provider interface, re-timing the whole
                         frozen transcript only when the forced aligner fails validation
    WHOLE-VOD (kept)     vod_processor: transcribe_accurate_text, transcribe_word_timing,
                         build_provisional_alignment / repair_word_timeline, process_vod;
                         the scouting transcript never becomes the published captions
"""
