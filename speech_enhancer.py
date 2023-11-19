from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement')

# for custom file, change path
est_sources = model.separate_file(path="ouptut.wav") 

torchaudio.save("my_voice_test.wav", est_sources[:, :, 0].detach().cpu(), 16000)