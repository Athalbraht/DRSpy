import xml.etree.ElementTree as ET
import json
import pandas as pd
from os import sep


class DataExtractor(object):
    """Used to extract data from XML file from oscilloscope to json or csv format"""

    def __init__(self):
        """init"""
        ### File paths
        self.source_path = ''
        self.destination_path = ''
        ### Verbosity
        self.verbose = False
        ### XML data
        self.xml_tree = None
        self.root = None
        ### Output data
        self.data = []
        ### VARS
        self.sep = sep

    def set_source_path(self, source: str):
        """sets source path to extract data"""
        self.verbose_msg(f'Previous source path was: {self.source_path}')
        self.source_path = source
        self.verbose_msg(f'New source path is: {self.source_path}')

    def load_xml_file(self):
        """Loads xml file from source path"""
        self.verbose_msg(f'Parsing XML file')
        self.xml_tree = ET.parse(self.source_path)
        self.root = self.xml_tree.getroot()
        self.verbose_msg(f'Done')

    def extract_data_to_dict(self):
        """Extracts data to json format"""
        self.verbose_msg(f'Extracting data from XML to dict')
        measurements = 0
        f_name = self.source_path.split(self.sep)[-1].split('.')[0]
        for child in self.root:
            measurements += 1
            tmp_dict = {}
            for data in child:
                if data.tag == 'Serial':
                    measurement_no = int(data.text)
                    tmp_dict['series'] = measurement_no
                    tmp_dict['file'] = f_name
                elif data.tag == 'Time':
                    tmp_dict['time'] = data.text
                elif data.tag == 'HUnit':
                    tmp_dict['unit_h'] = data.text
                elif data.tag == 'VUnit':
                    tmp_dict['unit_v'] = data.text
                elif data.tag == 'Board_2816':
                    ch1_data_x = []
                    ch1_data_y = []
                    ch2_data_x = []
                    ch2_data_y = []
                    for data2 in data:
                        if data2.tag == 'Trigger_Cell':
                            tmp_dict['trigger_cell'] = int(data2.text)
                        elif data2.tag == 'Scaler0':
                            tmp_dict['scaler0'] = int(data2.text)
                        elif data2.tag == 'Scaler1':
                            tmp_dict['scaler1'] = int(data2.text)
                        elif data2.tag == 'CHN1':
                            for row in data2:
                                x = float(row.text.split(',')[0])
                                y = float(row.text.split(',')[1])
                                ch1_data_x.append(x)
                                ch1_data_y.append(y)
                        elif data2.tag == 'CHN2':
                            for row in data2:
                                x = float(row.text.split(',')[0])
                                y = float(row.text.split(',')[1])
                                ch2_data_x.append(x)
                                ch2_data_y.append(y)
                    tmp_dict['ch1_x'] = ch1_data_x
                    tmp_dict['ch1_y'] = ch1_data_y
                    tmp_dict['ch2_x'] = ch2_data_x
                    tmp_dict['ch2_y'] = ch2_data_y
            self.data.append(tmp_dict)
        self.verbose_msg(f'Done\nNumber of measurements in file {measurements}')

    def reset_data(self):
        """Resets data"""
        self.data = []

    def save_as_json(self):
        """Saves data as json
        *adds json to path"""
        self.verbose_msg('Saving data')
        destination_path_json = self.destination_path
        with open(destination_path_json, 'w') as file:
            # json.dump(self.data, file, indent=4)
            json.dump(self.data, file)
        self.verbose_msg(f'Data saved to {destination_path_json}')

    def create_dst_path(self, file_type: str):
        """Creates destination path for all data
        :param file_type: Type of output file ex. json, csv
        :type file_type: str"""
        root_path = self.sep.join(self.source_path.split(self.sep)[:-2])
        self.extension = self.source_path.split(self.sep)[-1].split('.')[-1]
        f_name = self.source_path.split(self.sep)[-1].split('.')[0]
        self.destination_path = f'{root_path}{self.sep}extracted_data{self.sep}{file_type}{self.sep}{f_name}.{file_type}'

    def save_as_csv(self):
        """Saves data as csv
        *adds csv to path"""
        self.verbose_msg('Saving data')
        destination_path_csv = self.destination_path
        df = pd.DataFrame(data=self.data)
        # noinspection PyTypeChecker
        df.to_csv(destination_path_csv, index=False)
        self.verbose_msg(f'Data saved to {destination_path_csv}')

    def verbose_msg(self, msg: str):
        """For printing messages"""
        if self.verbose is True:
            print(msg)

    def set_verbose(self):
        """sets verbose mode"""
        self.verbose = True
