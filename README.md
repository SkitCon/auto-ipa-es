# auto-ipa-es
A model for automatic transcription of Spanish audio to segment IPA transcription aligned in a TextGrid file to facilitate manual phonemic transcription.

### version 1.0.1

# Table of Contents
1. [Description](#description)
2. [Steps to Run](#steps-to-run)
3. [Credits](#credits)

# Description

This is a Audio -> Aligned IPA transcription tool for Spanish meant to facilitate manual transcription. The ultimate goal of this project is to facilitate the transcription of a dataset of Spanish speech into IPA, accounting for pronunciation mistakes and slurs for fine-tuning of a final direct Speech -> IPA model. This model runs Audio -> Text -> IPA -> Aligned IPA, meaning that the generated IPA is based on the text, not the audio. An acceptable model does not currently exist which can generate IPA from audio directly. Sound transcription is highly error-prone without an underlying language model. Therefore, the goal is to create a model which can discern segment anomalies from an expected token + audio which is used to create the edited IPA transcription. In order to do this, a large corpus of speech is required where these anomalies are annotated, which is what this project facilitates.

The aligned output is in TextGrid format for input into many transcription platforms (such as Praat).

# Steps to Run

requirements.txt:
```
pip install numpy==1.26.4
pip install torch==2.4.0
pip install transformers==4.45.1
pip install librosa==0.10.2
```

1. Download the MFA accoustic model for Spanish from [here](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-spanish_mfa-v2.0.0)
2. Download the pre-trained grapheme to phoneme FST from [here](https://github.com/uiuc-sst/g2ps/blob/master/models/spanish_4_3_2.fst.gz), kindly pre-trained by the team of LanguageNet.
3. Install MFA and dependencies:
```
pip install montreal-forced-aligner
sudo apt install ffmpeg sox
```

4. Install Phonetisaurus using instructions from [here](https://github.com/AdolfVonKleist/Phonetisaurus) to install with Python3 bindings.
5. Run with `python3 transcribe.py <audio file>` (Note: the first time you run this, you will need to install whisper-large-v3 if you do not already have it which is a 3GB download)

# Credits

```
@techreport{mfa_spanish_mfa_dictionary_2022,
	author={McAuliffe, Michael and Sonderegger, Morgan},
	title={Spanish MFA dictionary v2.0.0},
	address={\url{https://mfa-models.readthedocs.io/pronunciation dictionary/Spanish/Spanish MFA dictionary v2_0_0.html}},
	year={2022},
	month={Mar},
}
```