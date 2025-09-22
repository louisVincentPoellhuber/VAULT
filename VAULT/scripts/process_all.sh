# Wikipedia Corpus
echo "Processing Wikipedia corpus."
bash preprocessing/extract_wikipedia.sh

echo "Processing HotPotQA."
python preprocessing/hotpotqa.py 

echo "Processing NQ."
python preprocessing/nq.py 

echo "Processing WikIR."
python preprocessing/wikir.py 

echo "Processing CORD19." 
python preprocessing/cord19.py

echo "Processing Doris-MAE." 
python preprocessing/doris_mae.py
