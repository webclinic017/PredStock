#!/bin/sh
apt update
apt upgrade -y
rm /home/toshi/PROJECTS/PredStock/data_j.xls
wget https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls -P /home/toshi/PROJECTS/PredStock
chown -R toshi /home/toshi/PROJECTS/PredStock/data_j.xls

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/toshi/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/toshi/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/toshi/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/toshi/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ag2
ipython /home/toshi/PROJECTS/PredStock/script/04_EDINET2.py
ipython /home/toshi/PROJECTS/PredStock/script/04_XBRL.py
chown -R toshi /home/toshi/PROJECTS/PredStock/XBRL/CSV/*
chown -R toshi /home/toshi/PROJECTS/PredStock/XBRL/ZIP_GET_CSV/*
chown -R toshi /home/toshi/PROJECTS/PredStock/XBRL/ZIP_NOT_CSV/*
ipython /home/toshi/PROJECTS/PredStock/script/00_SCRAPER14.py
ipython /home/toshi/PROJECTS/PredStock/script/00_SCRAPER14.py
chown -R toshi /home/toshi/PROJECTS/PredStock/00-JPRAW/*
chown -R toshi /home/toshi/PROJECTS/PredStock/00-JPRAW/NEW300/*
ipython /home/toshi/PROJECTS/PredStock/script/05_PRE.py
ipython /home/toshi/PROJECTS/PredStock/script/06_PREDICT.py
# ipython /home/toshi/PROJECTS/PredStock/script/03_DATA_TO_MODEL.py
# chown -R toshi /home/toshi/PROJECTS/PredStock/AutogluonModels/*
rsync /home/toshi/PROJECTS/PredStock/ /mnt/qnap/home/PROJECTS/PredStock/ -avh --delete
ipython /home/toshi/PROJECTS/PredStock/script/99_SHUTDOWN.py
