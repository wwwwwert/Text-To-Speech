pip install -r requirements.txt

# download model
wget "https://disk.yandex.ru/d/9abt9aZHPYyEfg" -O best_model.zip
unzip best_model.zip
rm best_model.zip

# get WaveGlove converter
cd waveglov
git clone https://github.com/xcmyz/FastSpeech.git
mv -v FastSpeech/text .
mv -v FastSpeech/audio .
mv -v FastSpeech/waveglow/* waveglow/
mv -v FastSpeech/utils.py .
mv -v FastSpeech/glow.py .
cp glow.py ../
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
rm -rf FastSpeech
cd ..

# download preprocessed data
mkdir data/datasets/ljspeech
cd data/datasets/ljspeech

# dataset
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null

# texts for audios
gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx

# mels
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz >> /dev/null

#download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null

cd ../../..
