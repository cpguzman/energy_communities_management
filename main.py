import json
with open('input_data_static.json') as json_file:
    par = json.load(json_file)
print("Static Data", par)
print("fin")