QB Passing EPA Analysis


By Eamon Mukhopadhyay


This project essentially analyzes the weekly quarterback passing data from 2024 and builds three model views: a cleaned broad model, an actionable subset model, and an efficiency sub model. It also exports a corr matrix, feature importance chart, and binning analysis visuals. The target variable used in workflow is passing_epa. Dataset by nflverse, via https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_2024.csv.

__________________________________
here are the two files needed in the same folder:

1. build_qb_analysis_corrected.py

2. nfl_qb_weekly_model_dataset_2024_flat.xlsx

___________________________________
HOW TO RUN: ----------------------------

Run from Terminal, Command Prompt or PowerShell in that folder.

1. Install packages:

pip install pandas openpyxl numpy matplotlib scikit-learn

2. Run the script:

python build_qb_analysis_corrected.py "nfl_qb_weekly_model_dataset_2024_flat.xlsx"

The output files will be saved in a folder named:

qb_model_outputs

That folder will contain the generated PNG charts and CSV output tables.
