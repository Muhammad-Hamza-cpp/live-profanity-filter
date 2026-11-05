# Related work and usable datasets

A literature and dataset search for a possible paper on the mute detector ([mute-detector.md](mute-detector.md)), done 2026-11-05. It covers web searches and abstracts, not a full reading of every paper. Licences were checked against dataset cards and official pages where the search reached them, and **must be re-checked at the source before use**. This is not legal advice.

---

## 1. Closest prior work

### Detecting profanity or abuse in speech

| Work | What it does | How we differ |
|---|---|---|
| [Spectrogram-Based Classification of Spoken Foul Language Using Deep CNN](https://ieeexplore.ieee.org/document/9287133/) (IEEE, 2020) | Classifies short clips as foul or normal, and into 10 curse-word classes. ResNet50 performed best. | Offline, clip-level. No streaming, no latency budget, no localisation inside a stream. |
| [Fast Spoken Foul Language Recognition with End-to-End DNNs](https://www.researchgate.net/publication/348666649_Design_and_Implementation_of_Fast_Spoken_Foul_Language_Recognition_with_Different_End-To-End_Deep_Neural_Network_Architectures) (Wazir et al., *Sensors* 2021) | CNN and RNN models on MFCC or mel features for spoken foul language. | Clip-level classification of isolated words. |
| [Deep Learning-Based Detection of Inappropriate Speech Content for Film Censorship](https://ieeexplore.ieee.org/document/9900333/) (IEEE, 2022) | CNNs that detect and **localise** foul language in film audio (macro AUC 93.85%). Reports that the acoustic approach beat an ASR pipeline. | **Closest academic work.** Offline film post-production with whole-file access. We decide under a 0.4 s live budget, from partially heard words. |
| [ADIMA: Abuse Detection In Multilingual Audio](https://arxiv.org/abs/2202.07991) (ShareChat, 2022) | 11,775 clips, about 65 h, 10 Indic languages. Clip-level abusive vs. not. Argues that ASR-first is slow and misses profanity. | Clip-level and non-English. It supports our "no transcript" motivation. |
| [DeToxy](https://arxiv.org/abs/2110.07592) (Ghosh et al., 2021/22) | English speech-toxicity dataset assembled from existing open corpora; utterance-level labels. Points out the keyword bias of text-based methods. | Utterance-level toxicity rather than word-level muting. A possible source of real curse-word audio, licensed per component corpus. |
| [MuTox](https://arxiv.org/abs/2401.05060) (Meta, Findings of ACL 2024) | Audio toxicity dataset in 30 languages (about 20k English/Spanish utterances) and a zero-shot classifier on SONAR embeddings. | Utterance-level and not streaming. A strong baseline to compare against on real speech. |
| [Profanity Detection from Audio Recordings Using NLP Techniques](https://link.springer.com/chapter/10.1007/978-981-99-9811-1_32) (Springer, 2024) | ASR followed by text classifiers. | The transcribe-then-match approach we argue against (our Whisper baseline). |

### Keyword spotting: latency, early decisions, labelling
- [Efficient keyword spotting using dilated convolutions and gating](https://arxiv.org/pdf/1811.07684) (Snips, 2018). Labelling *all* keyword frames makes a model fire as soon as the keyword starts, while end-of-keyword labels make it fire once it has enough context. **This is directly related to our v1 → v2 finding**: labelling frames not yet heard taught v1 to guess from context. Cite it, and frame our "heard-only" labelling as the streaming-muting version of the same question.
- [Latency Control for Keyword Spotting](https://arxiv.org/pdf/2206.07261) (2022). Trades detection latency against accuracy without knowing where the keyword ends. It's the closest thing to our latency/recall framing.
- Streaming CTC keyword spotting: [Contextual Biasing for Streaming ASR via CTC-based Word Spotting](https://arxiv.org/abs/2605.18222) (2026), [MFA-KWS](https://arxiv.org/abs/2505.19577) (2025), [Synaspot](https://arxiv.org/abs/2512.15124), and [Online KWS with a character-level RNN](https://arxiv.org/pdf/1512.08903) (2015). These are **baselines to compare against**: streaming KWS with the curse list as keywords.
- [Advances in Small-Footprint Keyword Spotting (review)](https://arxiv.org/html/2506.11169v1) (2025), and the curated list [awesome-keyword-spotting](https://github.com/zycv/awesome-keyword-spotting).

### Training keyword spotting on synthetic (TTS) speech: the main risk for us
- [Utilizing TTS Synthesized Data for Efficient Development of KWS](https://arxiv.org/abs/2407.18879) (Google, 2024). **Training on TTS alone did badly on real speech** (FRR about 46% vs 3% for real data). Adding real negatives helped a lot, and even about 2k real utterances closed most of the gap.
- [Adversarial training of KWS to minimize TTS data overfitting](https://arxiv.org/abs/2408.10463) (2024). Models latch onto TTS artefacts. An adversarial real/synthetic classifier gave up to 8% better accuracy.
- [Synth4Kws](https://arxiv.org/abs/2407.16840) (Interspeech SynData4GenAI 2024). More phrase and speaker diversity in TTS keeps improving results.
- [Training keyword spotters with limited and synthesized speech data](https://arxiv.org/abs/2002.01322) (Google, 2020). With a pre-trained speech embedding, training on synthetic speech alone matched about 500 real examples per keyword. **This is the same recipe as ours** (frozen encoder plus a small head).

These papers predict our weak point (synthetic-only training) and our fix (real negatives such as game audio, plus some real positives).

### Products and patents (prior art, not peer-reviewed)
- **Intel Bleep** (with Spirit AI, announced 2021): a Windows voice-chat filter with sliders for nine abuse categories and an on/off switch for the N-word. It hooks Windows audio, as our app does. No method or latency was published that we found. Coverage: [PC Gamer](https://www.pcgamer.com/intel-will-launch-its-anti-toxicity-voice-chat-ai-in-beta-later-this-year), [The Batch](https://www.deeplearning.ai/the-batch/news-haters-gonna-mute), [Voicebot](https://voicebot.ai/2021/03/26/intel-is-developing-a-real-time-language-filter-for-online-games).
- **Roblox patents, "Artificial latency for moderating voice communication":** [US 12,027,177 B2](https://patents.google.com/patent/US12027177) (filed 2022-09-08, granted 2024-07-02) and continuation [US 12,283,288](https://patents.justia.com/patent/12283288). An ML model scores the toxicity of an audio stream and the system **adds a delay based on that score**. The description also covers replacing the toxic part with noise or silence before transmitting it, and muting until a tirade ends. **We have not read the claims.** The delay-then-mute idea is protected prior art at the system level. Our contribution has to be the detector, its latency/recall/collateral behaviour, and the training method, not "buffer and mute". **Anyone using this commercially should have the claims reviewed by a patent professional.**
- **Modulate ToxMod** (voice-chat moderation used in Call of Duty): a vendor product. Modulate reports joint research with Caltech and Activision on toxicity in voice chat ([blog](https://modulate.ai/blog/caltech-activision-research-new-insights-into-toxicity-in-games)). It focuses on moderation, meaning reporting, rather than muting before playback.

### What looks new (based on this search; it doesn't prove novelty)
1. **Live, word-level muting of profanity under a fixed delay budget**, evaluated as a three-way trade-off between latency, recall and collateral (curse words silenced vs. clean words clipped). Prior academic work is offline or clip-level; products publish no methods or numbers.
2. **Heard-only frame labels** for streaming detection, plus the finding that templated synthetic data makes a detector **guess from context** (69% of v1's false mutes fired within 0.1 s). Context-matched negatives fix it.
3. **A frozen speech encoder with a tiny head, trained on TTS with exact phoneme alignments**: 0.93M trained parameters, 20 minutes of training on a laptop, per-group outputs that can be switched at runtime.
4. **Engineering:** encoder and head captured as one CUDA graph, 12 ms per check on a laptop GPU, measured in a real Windows audio path.

What the paper must add to be credible: **a real-speech test set**, **streaming KWS and MuTox-style baselines**, **curves across thresholds and delays**, and **multiple seeds**.

---

## 2. Data we can train on (licences)

**Rules of thumb:**
- **CC0 and CC BY** are fine for research and commercial use, with attribution for CC BY.
- **CC BY-SA** is fine but share-alike; check what that means for released weights.
- **NC (non-commercial)** is research only. A model trained on it should be treated as research-only too.

### Real speech, likely to contain swearing (positives)

| Dataset | Size | Licence | Notes |
|---|---|---|---|
| [YODAS / YODAS2](https://arxiv.org/abs/2406.00899) | 500k+ h, 100+ languages; 86k h with manual transcripts | Videos chosen because they carry Creative Commons licences; the paper's table says **CC BY 3.0** (check the HF card) | YouTube vlogs, gaming and talk content, so **the best bet for real spontaneous swearing**. Search the transcripts for curse words, fetch only those segments, then force-align. |
| [Emilia-YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 114k h | **CC BY 4.0** (the card's body text) | A cleaned, segmented subset of YODAS2. The **original Emilia** (101k h) is **CC BY-NC**, so avoid it for anything commercial. |
| [The People's Speech](https://arxiv.org/abs/2111.09344) (MLCommons) | ~30k h English | **CC BY / CC BY-SA** ("clean" and "dirty" subsets; per-work attribution CSV) | archive.org audio: talks, podcasts, radio. Transcripts are searchable. Already in our next-steps plan. |
| [DeToxy](https://arxiv.org/abs/2110.07592) | Large, assembled from open corpora | Per component corpus | Utterances already labelled toxic, a shortcut to real curse words. Check each source corpus's licence. |
| [MuTox](https://arxiv.org/abs/2401.05060) | ~20k En/Es utterances | **Check the release** | Mainly an evaluation set or baseline. |

### Real speech for negatives and diversity (clean, permissive)

| Dataset | Licence | Use |
|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/download) | **CC BY 4.0** (signals and transcripts) | Conversational and far-field speech, overlap, laughter: hard negatives |
| [Earnings-22](https://huggingface.co/datasets/esc-benchmark/esc-datasets) | **CC BY-SA 4.0** | Accented business English |
| LibriSpeech, [Multilingual LibriSpeech](https://arxiv.org/abs/2012.03411) | **CC BY 4.0** | Read speech; MLS is 50k h across 8 languages (covers the **Spanish/French game dialogue** that tripped v3) |
| Common Voice; [Common Voice Spontaneous Speech](https://datacollective.mozillafoundation.org/datasets/cmihqzerk023co20749miafhq) | **CC0** (English Spontaneous 1.0: ~9 h, 148 speakers, Nov 2025) | Accents and spontaneous English; some per-language listings add "don't re-share" terms, so check each |
| VoxPopuli | CC0 (check the card) | Multilingual parliament speech: non-English negatives |

### Non-speech: game-like sound effects, music, noise

| Dataset | Licence | Use |
|---|---|---|
| [MUSAN](https://openslr.org/17/) (OpenSLR 17) | **CC BY 4.0**; the curators say they excluded content that forbids commercial use | Noise, music, babble. **The obvious permissive replacement for ESC-50.** |
| [FSD50K](https://zenodo.org/record/4060432) | Dataset CC BY 4.0, but **clips are individually CC0 / CC BY / CC BY-NC / Sampling+** | 51k clips, 200 AudioSet classes, including weapons and explosions. **Keep only CC0 and CC BY clips** for commercial use, and check the gunshot class name in `vocabulary.csv`. |
| [ESC-50](https://github.com/karoldvl/ESC-50) | **CC BY-NC** (the ESC-10 subset is CC BY) | ⚠️ **v3 was trained on ESC-50 folds 1–4**, so treat `detector-v3.pt` as **research-only** until it's retrained on MUSAN and CC0/CC BY FSD50K clips. |

### Data that cannot be redistributed
- **Your Call of Duty recordings** are copyrighted game audio. They're fine for personal training and evaluation on your own machine, but they **can't be published as a dataset**, and that limits what a paper can share. Use them as a private evaluation set, and say so in the paper.
- **GigaSpeech, SPGISpeech, TED-LIUM, Switchboard/Fisher (LDC):** restricted or non-commercial licences, or paid. (These come from memory, not this search; verify before excluding them.)

---

## 3. Suggested plan

1. **Real-speech test set (most important):**
   - Search the YODAS2 manual-transcript subset and People's Speech transcripts for words in our lexicon.
   - Download only the matching segments.
   - Force-align the words (CTC alignment with Parakeet or MMS).
   - Hand-check 300–500 curse words by ear.
   - Add an equal amount of clean spontaneous speech from AMI and Common Voice Spontaneous.
   - This set is publishable under CC BY / CC BY-SA, with attribution.
2. **Retrain with permissive data only (v4):**
   - Replace ESC-50 with MUSAN plus CC0/CC BY FSD50K (gunshot, explosion, vehicle, music classes).
   - Add MLS or VoxPopuli Spanish and French as non-English negatives.
   - Add some real positives from YODAS2 and People's Speech (the Google TTS-KWS paper suggests a few thousand real utterances close most of the gap).
3. **Baselines for the paper:**
   - Whisper with a word list (our app's default)
   - Parakeet CTC gap-fill
   - one streaming CTC keyword spotter
   - MuTox (utterance-level, to show why word-level timing matters)
4. **Report:**
   - curves of delay vs. recall vs. clean-word collateral
   - three training seeds
   - ablations: heard-only labels, context negatives, sound-effect negatives, confirm-K, pre-pad, speech gate
   - checks on the laptop GPU
