# Most code made by Aaron (Yinghao) Li from https://github.com/yl4579/StyleTTS2/.
import torch
import random
import random
import time
import librosa
import numpy as np
import torch
import torchaudio
import yaml
import nltk 
from nltk.tokenize import word_tokenize
import os
import phonemizer
from scipy.io.wavfile import write

from styletts2_inferencer.models import *
from styletts2_inferencer.text_utils import TextCleaner
from styletts2_inferencer.utils import *
from styletts2_inferencer.Utils.PLBERT.util import load_plbert


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


class StyleTTS2Inferencer:
    def __init__(self, 
        model_dir="StyleTTS2-LJSpeech/Models/LJSpeech/",
        config_file="config.yml",
        model_file="epoch_2nd_00100.pth",
        device="cuda",
    ):
        print("Initializing StyleTTS2 inferencer..")
        self.model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_dir)
        self.model_path = os.path.join(self.model_dir, model_file)
        self.config_path = os.path.join(self.model_dir, config_file)
        self.device = device

        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        np.random.seed(0)
        self.textcleaner = TextCleaner()
        nltk.download('punkt')

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        # load config and download model if not present
        self.config = self.load_config()

        self.config["ASR_path"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["ASR_path"])
        self.config["ASR_config"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["ASR_config"])
        self.config["F0_path"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["F0_path"])
        self.config["PLBERT_dir"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["PLBERT_dir"])

        # load pretrained ASR model
        ASR_config = self.config.get("ASR_config", False)
        ASR_path = self.config.get("ASR_path", False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = self.config.get("F0_path", False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = self.config.get("PLBERT_dir", False)
        plbert = load_plbert(BERT_path)

        self.model = build_model(
            recursive_munch(self.config["model_params"]),
            text_aligner,
            pitch_extractor,
            plbert,
        )
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load(self.model_path, map_location="cpu")
        params = params_whole["net"]

        for key in self.model:
            if key in params:
                print("%s loaded" % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict

                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], self.model[key])
        _ = [self.model[key].eval() for key in self.model]

        from styletts2_inferencer.Modules.diffusion.sampler import (
            ADPM2Sampler,
            DiffusionSampler,
            KarrasSchedule,
        )

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

    def load_config(self):
        # check if path exists
        if not os.path.isdir(self.model_dir):
            print("Config file not found")
            # check if model files exist and download if not
            if not os.path.exists(self.config_path):
                print("Downloading StyleTTS2 model from Huggingface..")
                from git import Repo
                Repo.clone_from(url="https://huggingface.co/yl4579/StyleTTS2-LJSpeech", 
                                to_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "StyleTTS2-LJSpeech"))
                

        config = yaml.safe_load(open(self.config_path))
        return config

    def synthesize_speech_long(self, text, filepath):
        sentences = text.split(".")  # simple split by dot
        wavs = []
        s_prev = None
        for text in sentences:
            if text.strip() == "":
                continue
            text += "."  # add it back
            noise = torch.randn(1, 1, 256).to(self.device)
            wav, s_prev = self.LFinference(
                text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5
            )
            wavs.append(wav)
        self.write_to_file(np.concatenate(wavs), rate=24000, filepath=filepath)

    def synthesize_speech(self, text, filepath):
        if len(text.split(".")) > 1:
            self.synthesize_speech_long(text, filepath)
        else:
            start = time.time()
            noise = torch.randn(1, 1, 256).to(self.device)
            wav = self.inference(text, noise, diffusion_steps=5, embedding_scale=1)
            rtf = (time.time() - start) / (len(wav) / 24000)
            print(f"RTF = {rtf:5f}")
            self.write_to_file(wav, rate=24000, filepath=filepath)

    def write_to_file(self, wav, rate=24000, filepath="output.wav"):
        write(filepath, rate, wav)
        print(f"Written output to file {filepath}")

    def tts_to_file(self, text, filepath):
        self.synthesize_speech(text, filepath)


    def inference(self, text, noise, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', "")
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)

        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0),
            )

        return out.squeeze().cpu().numpy()


    def LFinference(self, text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', "")
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)

        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = alpha * s_prev + (1 - alpha) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0),
            )

        return out.squeeze().cpu().numpy(), s_pred
    
    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, ref_dicts):
        reference_embeddings = {}
        for key, path in ref_dicts.items():
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)
            mel_tensor = self.preprocess(audio).to(self.device)

            with torch.no_grad():
                ref = self.model.style_encoder(mel_tensor.unsqueeze(1))
            reference_embeddings[key] = (ref.squeeze(1), audio)

        return reference_embeddings


def main():
    tts = StyleTTS2Inferencer()
    tts.tts_to_file("Hello! How are you doing?", "/home/plip/Downloads/test.wav")

if __name__ == "__main__":
    main()
