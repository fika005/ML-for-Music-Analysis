The steps to prepare the data and train the model:
- create a folder `data` by
    - ` mkdir data data_ogg`
- Run `download_free_musics.py` by
    - `python download_free_musics.py meta_data.csv data`
- Run `convert_files` notebook
- Clone SALAMI repo:
    -  `git clone https://github.com/DDMAL/salami-data-public.git`
- Run `extract_labels` notebook
- Run `python run.py`