from helper_functions import load_skeletal_data, get_all_frames_by_id

csv_file = r'C:\Users\User\Downloads\ENCM 509 Project\Anthropometric-Measurement-analysis-in-Python\Skeletal Coordinates.csv'
df = load_skeletal_data(csv_file)
x = get_all_frames_by_id(df, person_id='1')
print(x)


