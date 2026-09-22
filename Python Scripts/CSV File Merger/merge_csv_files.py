import glob
import pandas as pd

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
output_filename = 'combined_csv.csv'

if output_filename in all_filenames:
    all_filenames.remove(output_filename)

if not all_filenames:
    raise FileNotFoundError('No source CSV files were found in the current directory.')

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
combined_csv.to_csv(output_filename, index=False, encoding='utf-8-sig')
