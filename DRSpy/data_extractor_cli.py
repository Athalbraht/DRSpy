import argparse
from data_extractor import DataExtractor
from os import path, listdir, sep
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='Turns verbose output', action="store_true")
parser.add_argument('-s', '--source', help='Sets source file or directory', type=str)
parser.add_argument('-d', '--destination', help='Sets destination file or directory', type=str)
parser.add_argument('-t', '--type', help='set final file format json or csv', type=str)
args = parser.parse_args()
extractor = DataExtractor()
t1 = time()
if args.verbose:
    extractor.set_verbose()
if path.isdir(args.source):
    files = listdir(args.source)
    files_to_do = len(files)
    for file_name in files:
        print(f'Extracting data from {file_name}')
        extractor.source_path = (args.source + sep + file_name)
        extractor.create_dst_path(args.type)
        extractor.load_xml_file()
        extractor.extract_data_to_dict()
        if args.type == 'csv':
            extractor.save_as_csv()
        elif args.type == 'json':
            extractor.save_as_json()
        else:
            print('Unknown file type')
        extractor.reset_data()
        files_to_do -= 1
        if files_to_do > 0:
            print(f'Remaining: {files_to_do}')
        else:
            print('Done')
else:
    extractor.set_source_path(args.source)
    extractor.destination_path = args.destination
    extractor.load_xml_file()
    if args.type == 'csv':
        extractor.save_as_csv()
    elif args.type == 'json':
        extractor.save_as_json()
    else:
        print('Unknown file type')
print(f'All done\nTime Taken: {time() - t1}')
