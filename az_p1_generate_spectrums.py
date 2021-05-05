from library.generate_spectrums_and_save_them import generate_spectrums


""" No_Attack = True    >>>    There is no attack from any eavesdropper

    No_Attack = False   >>>    An eavesdropper is attacking the network """


# Firstly, we generate spectrums in the case of non-eavesdropping attack
# The data is stored in 'input/MUSIC_spectrums_label_0.csv'
generate_spectrums(No_Attack=True)

# Secondly, we generate spectrums in the case of an eavesdropping attack
# The data is stored in 'input/MUSIC_spectrums_label_1.csv'
generate_spectrums(No_Attack=False)

# =============================================================================
""" It's time to merge 2 csv files into 1 csv file

link 1: https://stackoverflow.com/questions/2512386/how-to-merge-200-csv-files-in-python

"""
# the csv files are MUSIC_spectrums_label_0.csv and MUSIC_spectrums_label_1.csv
# the merged csv file is spectrum.csv


fout = open('input/spectrum.csv', 'a')
num_files = 2
for num in range(2):
    for line in open('input/MUSIC_spectrums_label_' + str(num) + '.csv'):
        fout.write(line)
fout.close()
