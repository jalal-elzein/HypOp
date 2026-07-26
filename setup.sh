conda create --name hypop -y python=3.10
conda activate hypop

pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r dependency.txt
