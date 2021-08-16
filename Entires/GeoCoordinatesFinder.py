import regex
from collections import namedtuple
import numpy as np
import pandas as pd
import Settings, Functions


PartsAddress = namedtuple('PartsAddress', 'street_type, street_name, house_number', defaults=('', '', 0))


class MoscowCharacteristics:
    MIN_STREET_LENGHT = 4
    def __init__(self, geo_base):
        self.types_streets_and_abbreviation= self._get_types_streets_and_abbreviation()
        self.types_streets_by_abbreviation = self._get_street_types_by_abbreviation()
        self.modifiers_streets = self._get_modifiers_streets()
        self.regions_of_Moscow = self._get_regions_of_Moscow()
        self._set_statistics_ending_street(geo_base)
        
    def _get_types_streets_and_abbreviation(self):
        return {'переулок':    ['пер'],  
                 'проспект':    ['пр-т', 'пр-кт', 'п-т', 'п-кт', 'п-кт', 'просп', 'прос', 'пркт', 'пр'],
                 'шоссе':       ['ш', 'шс', 'шосс'],
                 'проезд':      ['пр-д', 'п-д', 'пр'],
                 'набережная':  ['наб'], 
                 'тупик':       ['т-к', 'туп'], 
                 'бульвар':     ['бул', 'б-р', 'бульв', 'бр'],
                 'аллея':       ['алл', 'ал'], 
                 'квартал':     ['кв-л'], 
                 'микрорайон':  ['м-он', 'м-н','мон', 'мкр'],
                 'площадь':     ['пл', 'пл-дь'],
                 'улица':       ['ул']
                 }
    
    def _get_street_types_by_abbreviation(self):
        result = {}
        for street_type, abbreviations in self.types_streets_and_abbreviation.items():
            result[street_type] = [street_type]
            
            for abbreviation in abbreviations:
                if abbreviation in result:
                    result[abbreviation].append(street_type)
                else:
                    result[abbreviation] = [street_type]
                    
        return result
    
    @staticmethod
    def _get_modifiers_streets():
        modifiers = {}
        modifiers['бол'] =  ('большая', 'большой')
        modifiers['мал'] =  ('малая', 'малый')
        modifiers['б'] =    ('большая', 'большой')
        modifiers['м'] =    ('малая', 'малый')
        modifiers['вер'] =  ('верхняя', 'верхний')
        modifiers['верх'] = ('верхняя', 'верхний')
        modifiers['верхн'] =('верхняя', 'верхний')
        modifiers['ниж'] =  ('нижняя', 'нижний')
        modifiers['нижн'] = ('нижняя', 'нижний')
        modifiers['стар'] = ('старая', 'старый')
        modifiers['нов'] =  ('новая', 'новый')
        return modifiers
            
    def _get_regions_of_Moscow(self):
        with open(Settings.ROOT_DIR + '/Dictionaries/regions_of_Moscow.txt', 'r') as text_file:
            lines = text_file.readlines()
        lines = list(map(str.strip, lines))
        lines = list(map(str.lower, lines))
        return lines
    
    def _set_statistics_ending_street(self, geo_base):
        stat = geo_base.base_addresses.reset_index()
        stat['ending'] = stat['street_name'].apply(lambda row: row[-1])
        stat = stat.drop(['street_name', 'street_modifiers', 'latitude', 'longitude', 'house_number'], axis=1)
        stat = stat.value_counts()
        stat = stat.unstack(level=0)
        stat= stat.fillna(0)
        dict_stat = stat.to_dict(orient='index')
        
        sorted_stat = {}
        for letter, stat_for_letter in dict_stat.items():
            sorted_stat[letter] = [k for k, v in sorted(stat_for_letter.items(), key=lambda item: -item[1]) if v>0]
            
        self.statistic_of_ending_street = sorted_stat
        
    def get_types_streets(self, street_name):
        last_letter = street_name[-1]
        if last_letter in self.statistic_of_ending_street:
            return self.statistic_of_ending_street[last_letter]
        else:
            return []
   
   
class GeoBase:
    def __init__(self):
        self.base_addresses = self._get_base_addresses_and_coordinates()
            
    def _get_base_addresses_and_coordinates(self):
        def set_values(street_type, street_name, street_modifier, house_number):
            street_types.append(street_type)
            street_names.append(street_name)
            street_modifiers.append(street_modifier)
            house_numbers.append(house_number)
            longitudes.append(row['longitude'])
            latitudes.append(row['latitude'])
            
        file_name = Settings.PATH_BASE_COORDINATES_OF_ADDRESSES
        dtype = [('street','U50'),('house','U10'),('latitude','float64'),('longitude','float64')]
        array = np.genfromtxt(file_name, dtype, delimiter=';', skip_header=1, skip_footer=1)
        
        types_of_street = ['переулок', 'проспект', 'шоссе', 'проезд', 'набережная', 'тупик', 'бульвар', 'аллея', 'квартал', 'микрорайон', 'площадь', 'улица']
        
        street_types = []
        street_names = []
        street_modifiers = []
        house_numbers = []
        latitudes = []
        longitudes = []
        
        ranks = set(['академика', 'генерала', 'героя', 'маршала', 'адмирала'])
        
        all_modifiers_streets = set()
        modifiers_streets = MoscowCharacteristics._get_modifiers_streets()
        for femine, masculine in modifiers_streets.values():
            all_modifiers_streets.add(femine)
            all_modifiers_streets.add(masculine)
        
        endings = [letter for letter in '0123456789абвгдежзийклмнопрстуфхцчшщыъьэюя']    
        endings = set(endings)
        
        for row in array:
            house = row['house']
            street = row['street']
            if (not house) or (not street):
                continue
            
            if house.startswith('вл'):
                house = house[2:]
                
            house_number = Functions.str_to_int(house)
            if house_number == 0:
                continue
            
            street_type = None
            street_name = None
            street = street.lower()
            street = street.replace('ё', 'е')
            
            code_symbol = ord(street[0])
            if code_symbol>=49 and code_symbol<=57:
                space = street.find(' ')
                number_street = street[:space]
                street = street[space+1:]
            else:
                number_street = ''
            
            street_type = 'улица'
            street_name = street        
            for type_of_street in types_of_street:
                left = street.find(type_of_street + ' ')
                if left >= 0:
                    street_type = type_of_street
                    street_name = street.replace(type_of_street + ' ', '')
                    break
                
                right = street.find(' ' + type_of_street)
                if right >= 0:
                    street_type = type_of_street
                    street_name = street.replace(' ' + type_of_street, '')
                    break
                
            if street_name is None or (not street_name):
                continue
            
            street_modifier = number_street
            for modifier in all_modifiers_streets:
                if street_name.find(modifier) != -1:
                    street_name = street_name.replace(modifier + ' ', '')
                    street_modifier = number_street or modifier
                    break
            
            street_name = street_name.replace('-', ' ')
            set_values(street_type, street_name, street_modifier, house_number)
            
            space = street_name.find(' ')
            if space > 0:
                first_word = street_name[:space]
                second_word = street_name[space+1:]
                
                street_name = second_word + ' ' + first_word
                set_values(street_type, street_name, street_modifier, house_number)
                
                if first_word in ranks:
                    set_values(street_type, second_word, street_modifier, house_number)
                
                elif second_word in ranks:
                    set_values(street_type, first_word, street_modifier, house_number)
                
        df = pd.DataFrame({'street_type':street_types, 'street_name':street_names, 'street_modifiers':street_modifiers, 'house_number':house_numbers, 'longitude':longitudes, 'latitude':latitudes})
        df.set_index(['street_type', 'street_name', 'street_modifiers', 'house_number'], inplace=True)
        df.sort_index(level=0, inplace=True)
            
        return df
    
    def find_coordinates(self, struct_address):
        def get_result(consider_house_number):
            if consider_house_number:
                search_fields = (street_type, street_name, street_modifier, house_number)
            else:
                search_fields = (street_type, street_name, street_modifier)
                
            if search_fields in self.base_addresses.index:
                return self.base_addresses.loc[search_fields]
            return None
        
        def get_near_house(houses, modulo, forward):
            if forward:
                start = 0
                step = 1
                end = len(houses)
            else:
                start = len(houses) - 1
                step = -1
                end = -1
                
            for index_row in range(start, end, step):
                house = houses.iloc[index_row]
                if house.name%2 == modulo:
                    return house.name, house.latitude, house.longitude
            return None
                
        if not struct_address:
            return None
        
        street_type = struct_address['street_type']
        street_name = struct_address['street_name'] 
        street_modifier = struct_address['street_modifier'] 
        house_number = struct_address['house_number']    
        
        #find with house number
        search_result = get_result(consider_house_number=True)
        if type(search_result) == pd.DataFrame:
            #get first row
            search_result = search_result.iloc[0]
        
        if type(search_result) == pd.Series:    
            return (search_result[0], search_result[1])
                    
        #find without house number
        search_result = get_result(consider_house_number=False)
        if type(search_result) == pd.DataFrame and len(search_result)>0:
            
            modulo = house_number%2
            left_houses = search_result[search_result.index<house_number]
            right_houses = search_result[search_result.index>house_number]
            
            left_house = get_near_house(left_houses, modulo, forward=False)
            right_house = get_near_house(right_houses, modulo, forward=True)
            if None not in [left_house, right_house]:
                left_number, left_latitude, left_longitude = left_house
                right_number, right_latitude, right_longitude = right_house
                
                k = (house_number-left_number) / (right_number-left_number)
                longitude = left_longitude + (right_longitude-left_longitude) * k
                latitude = left_latitude + (right_latitude-left_latitude) * k 
                
            else:
                #find center of street
                center = int(len(search_result) / 2)
                center_house = search_result.iloc[center]
                longitude = center_house.longitude 
                latitude = center_house.latitude
                
            return (longitude, latitude)
                    
        return None
 
      
class _AddressSplitter:
    def __init__(self, source_address, moscow_caracteristic):
        self.moscow_caracteristic = moscow_caracteristic
        self.address = source_address
        
        self._remove_administrative_districts()
        self._cast_lower_case()
        
    def _remove_administrative_districts(self):
        address = self.address
        address = address.replace('СЗАО', '')
        address = address.replace('ЮЗАО', '')
        address = address.replace('СВАО', '')
        address = address.replace('ЮВАО', '')
        address = address.replace('САО', '')
        address = address.replace('ВАО', '')
        address = address.replace('ЮАО', '')
        address = address.replace('ЗАО', '')
        address = address.replace('ЦАО', '')
        
        address = address.replace('ЗЕЛАО', '')
        address = address.replace('НАО', '')
        address = address.replace('ТАО', '')
        self.address = address
    
    def _cast_lower_case(self):
        self.address = self.address.lower()
            
    def positioned_by_type_address(self, type_address):
        if not type_address:
            return True
        
        address = self.address
        position = address.find(type_address)
        if position == -1:
            return False
        
        start_address = position + len(type_address)
        
        last_sep = 0
        lenght = len(address)
        separators = set(['-', ':', ' '])
        
        for i in range(5):
            if (start_address+i)<lenght and address[start_address+i] in separators:
                last_sep = start_address + i
            else:
                break
        
        if last_sep > 0:
            start_address = last_sep + 1
            
        self.address = address[start_address:]
        return True
        
    def this_is_Moscow(self):
        def find_name(delta, names):
            pattern_full = regex.compile(r'''(?<=^''' + delta + ''')''' + names + '''
                                    (?=[-,\.\s]+
                                    .*$)
                                    ''', regex.VERBOSE)
            if pattern_full.findall(searched):
                return True
            
        searched = self.address[:20]
        deltas = ['', '.*[-,\.\s]+']
        
        full_names = '(московская|область|край|республика){1,2}'
        for delta in deltas:
            if find_name(delta, full_names):
                return False  
        
        short_names = '(мо|обл|кр|респ){1,2}'
        for delta in deltas:
            if find_name(delta, short_names):
                return False 
        return True
    
    def _positioned_after_words_Moscow_and_Zelenograd(self):
        address = self.address
        lenght = len(address)    
        start_address = 0    
        separators = set([',', '.', ';', '-', ':', ' '])
        towns = ['москва г ', 'москва г.', 'москва г,','москва', ' м-ва г ', ' м-ва г.', ' м-ва г,', ' м-ва ', ' м. ']
        for town in towns:
            position = address.find(town)
            if position == -1:
                continue
            
            start_address = position + len(town)
            
            position_zelenograd = address.find('зеленоград', start_address)
            if position_zelenograd != -1:
                start_address = position_zelenograd + 10
            
            last_sep = 0        
            for i in range(5):
                if (start_address+i)<lenght and address[start_address+i] in separators:
                    last_sep = start_address + i
                else:
                    break
        
            if last_sep > 0:
                start_address = last_sep + 1      
            break
        self.address = address[start_address:]
    
    def _replace_nonstandard_symbols(self):
        def replace_symbols(symbols, desired_value):
            table_symbols = str.maketrans({k:desired_value for k in symbols})
            return address.translate(table_symbols)
        
        address = self.address    
        address = address.replace('ё', 'е')
        address = replace_symbols([';', '!', '?'], ',')
        address = replace_symbols(['"', "'", '№', '*'], '')
        
        address = address.replace(',,', ',')
        address = address.replace('..', '.')
        address = address.replace('--', '-')
        
        address = replace_symbols([':', "\t", '\n', '(', ')', '[', ']', '{', '}'], ' ')
        address = address.replace('  ', ' ')
        self.address = address
        
    def _remove_regions(self): 
        address = self.address
        
        modificator = None
        modificators = [' район ', ' район,', ',район ', ' р-он ', ' р-он,', ',р-он ', ' р-н ', ' р-н,', ',р-н ']
        for current_modificator in modificators:
            if address.find(current_modificator) != -1:
                modificator = current_modificator
                break
            
        if modificator is None:
            return address
        
        regions_of_Moscow = self.moscow_caracteristic.regions_of_Moscow        
        for region in regions_of_Moscow:
            if modificator[0] == ',':
                address = address.replace(modificator[1:] + region, '')
                
            elif modificator[-1] == ',':
                address = address.replace(region + modificator[:-1], '')
                
            else:
                address = address.replace(modificator[1:] + region, '')
                address = address.replace(region + modificator[:-1], '')
                
        self.address = address
    
    def _replace_abbrevations(self):
        def find_name(delta):
            pattern_full = regex.compile(r'''(?<=^''' + delta + ''')''' + shorts + '''
                                        \.*
                                        (?=([-,\s]+.*$))
                                        ''', regex.VERBOSE)
            return pattern_full.search(address)
        
        address = self.address    
        abbrevations = {'ак':'академика', 'ген':'генерала', 'гер':'героя', 'марш':'маршала', 'адм':'адмирала', 'косм': 'космонавта'}
        shorts = abbrevations.keys()
        shorts = '|'.join(shorts)
        shorts = '(' + shorts + '){1}'
        
        deltas = ['', '(.*[-,\.\s]+)']
        for delta in deltas:
            parsing_address = find_name(delta)
            if parsing_address is None:
                continue
            
            if delta:
                left = parsing_address[1]
                middle = parsing_address[2]
                right = parsing_address[3]
            else:
                left= ''
                middle = parsing_address[1]
                right = parsing_address[2]
                
            full_word = abbrevations[middle]
            address = left + full_word + right
            break
                
        self.address = address
            
    def _remove_leading_commas_and_points(self):
        address = self.address
        shift = -1
        for index in range(len(address)):
            if address[index] in [' ', ',', '.']:
                shift = index
            else:
                break
                
        self.address = address[shift+1:]
        
    def _remove_street_number(self):
        pattern = regex.compile(r'''([1-9]{1}|1{1}[0-9]{1}) #[1-19]
                                (\s?)
                                (-?)
                                (\s?)
                                ([яаыои]?) #first symbol of ending
                                (я|й){1} #second symbol of ending
                                ([ ,\.]{1})
                                ''', regex.VERBOSE)
        
        number_and_endings = pattern.findall(self.address)
        if not number_and_endings:
            return ''
        
        symbols = number_and_endings[0]
        street_for_replacing = ''
        for index in range(6):
            street_for_replacing += symbols[index]
        
        if symbols[6] == ' ':
            street_for_replacing += ' '
        else:
            street_for_replacing = ' ' + street_for_replacing
            
        self.address = self.address.replace(street_for_replacing, '')
        
        street_number = ''
        for index in [0, 2, 5]:
            street_number += symbols[index]
            
        return street_number
       
    @staticmethod
    def _get_pattern(searching_string):
        pattern = ''
        if type(searching_string) == list:
            for element in searching_string:
                pattern += element + '|'
        else:
            pattern = searching_string + '|'
        return pattern
    
    @staticmethod                    
    def _add_patterns(patterns, searching_strings, these_are_abbreviations):
        commom_part_pattern = '('
        for searching_string in searching_strings:
            commom_part_pattern += _AddressSplitter._get_pattern(searching_string)
        
        commom_part_pattern = commom_part_pattern[:-1]
        commom_part_pattern += '){1}'
        
        if these_are_abbreviations:
            right_splitter = '[\.\s]+'
        else:
            right_splitter = '\s+'
            
        pattern_left_edge = '(?<=^)'           + commom_part_pattern + '(?=' + right_splitter + '(.*)$)'            
        pattern_center =    '(?<=^(.*)\s+)'    + commom_part_pattern + '(?=' + right_splitter + '(.*)$)'
        pattern_right_edge ='(?<=^(.*)\s+)'    + commom_part_pattern + '(?=$)'
        
        patterns.append(('left', regex.compile(pattern_left_edge)))
        patterns.append(('center', regex.compile(pattern_center)))
        patterns.append(('right', regex.compile(pattern_right_edge)))
                
    def _split_streets_and_types_streets(self, str_with_street):
        comma = str_with_street.rfind(',')
        if comma != -1:
            str_with_street = str_with_street[comma+1:]
        
        str_with_street = str_with_street.strip()
        if len(str_with_street) < self.moscow_caracteristic.MIN_STREET_LENGHT:
            return set()
        
        patterns = []
        types_streets_and_abbreviation = self.moscow_caracteristic.types_streets_and_abbreviation
        self._add_patterns(patterns, types_streets_and_abbreviation.keys(), these_are_abbreviations=False)
        self._add_patterns(patterns, types_streets_and_abbreviation.values(), these_are_abbreviations=True)
        
        possible_splits = []
        types_streets_by_abbreviation = self.moscow_caracteristic.types_streets_by_abbreviation
        for where_street_type, pattern in patterns:
            result = pattern.findall(str_with_street)
            for parsed_address in result:
                streets = []
                if where_street_type == 'left':
                    streets.append(parsed_address[1])
                    street_type = parsed_address[0]
    
                elif where_street_type == 'center':
                    streets.append(parsed_address[0])
                    street_type = parsed_address[1]
                    streets.append(parsed_address[2])
                    
                elif where_street_type == 'right':
                    streets.append(parsed_address[0])
                    street_type = parsed_address[1]
                
                street_types = types_streets_by_abbreviation[street_type]
                for street_type in street_types:
                    for street_name in streets:
                        if len(street_name) < self.moscow_caracteristic.MIN_STREET_LENGHT:
                            continue
                        
                        street_name = street_name.replace('улица', '') 
                        street_name = street_name.replace('ул.', '')
                        street_name = street_name.replace('ул ', '')
                        street_name = street_name.replace('ул,', ',')
                        street_name = street_name.replace('  ', ' ')
                        street_name = street_name.strip()
                        possible_splits.append(PartsAddress(street_type, street_name))
            
        #add if no type of street
        if not possible_splits:
            street_types = self.moscow_caracteristic.get_types_streets(str_with_street)
            for street_type in street_types:    
                possible_splits.append(PartsAddress(street_type, str_with_street))

        return possible_splits   
 
    def _split_address_by_house(self):
        variants_parsed_address = []
        address = self.address
        
        pattern_split_by_house = regex.compile(r'''(?<=^(.+)[-,\.\s]+)
                                            (дом|д){1}
                                            (?=[-\.\s]*([0-9]+).*$)
                                            ''', regex.VERBOSE)
        
        parsed_address = pattern_split_by_house.findall(address)
        for parsed_address in parsed_address:
            house_number=int(parsed_address[2])
            
            possible_splits = self._split_streets_and_types_streets(parsed_address[0])
            for possible_split in possible_splits:
                parts_address = PartsAddress(possible_split.street_type, possible_split.street_name, house_number)
                variants_parsed_address.append(parts_address)
                
        return variants_parsed_address
    
    def _split_address_by_street_without_house(self):
        def get_left_edge_parsed_address(delta):
            positions = (0, 1, 2)
            pattern_type_left_edge = regex.compile(r'''(?<=^''' + delta + ''')''' + names + '''
                                                (?=[\.\s]+
                                                ([а-я\s-]{4,}) #name of street
                                                [-,\.\s\/\\\\]+
                                                ([0-9]+) #number of houses
                                                .*$)
                                                ''', regex.VERBOSE)
            variants_parsed_address = pattern_type_left_edge.findall(address)
            return variants_parsed_address, positions
        
        def get_right_edge_parsed_address():
            positions = (1, 0, 2)
            pattern_type_right_edge = regex.compile(r'''(?<=^(.*)[-\s]+)''' + names + '''
                                                (?=[-,\.\s\/\\\\]+
                                                ([0-9]+) #numbers of houses
                                                .*$)
                                                ''', regex.VERBOSE)
            variants_parsed_address = pattern_type_right_edge.findall(address)    
            return variants_parsed_address, positions
            
        def fill_variants_parsed_address(variants_parsed_address, positions, type_street_left_edge):
            for variant in variants_parsed_address:
                street_type = variant[positions[0]]
                street_name = variant[positions[1]]
                house_number = variant[positions[2]]
                
                if type_street_left_edge:
                    comma = street_name.find(',')
                    if comma != -1:
                        street_name = street_name[:comma-1]
                else:
                    comma = street_name.rfind(',')
                    if comma != -1:
                        street_name = street_name[comma+1:]
                        
                street_name = street_name.replace('улица', '') 
                street_name = street_name.replace('ул.', '')
                street_name = street_name.replace('ул ', '')
                street_name = street_name.replace('ул,', ',')
                street_name = street_name.replace('  ', ' ')
                street_name = street_name.strip()
                if len(street_name) < self.moscow_caracteristic.MIN_STREET_LENGHT:
                    continue

                house_number = int(house_number)
                
                types_streets_by_abbreviation = self.moscow_caracteristic.types_streets_by_abbreviation
                street_types = types_streets_by_abbreviation[street_type]
                for street_type in street_types:    
                    all_variants_parsed_address.append(PartsAddress(street_type, street_name, house_number))
            
        all_variants_parsed_address = []
        address = self.address
        types_streets_and_abbreviation = self.moscow_caracteristic.types_streets_and_abbreviation
        
        names = '('
        for full_name, abbreviations in types_streets_and_abbreviation.items():
            names += full_name + '|'
            for abbreviation in abbreviations:
                names += abbreviation + '|'
        names = names[:-1] + '){1}'
        
        variants_parsed_address, positions = get_left_edge_parsed_address('.*[-,\.\s]+')
        fill_variants_parsed_address(variants_parsed_address, positions, type_street_left_edge=True)
            
        variants_parsed_address, positions = get_left_edge_parsed_address('')
        fill_variants_parsed_address(variants_parsed_address, positions, type_street_left_edge=True)
        
        variants_parsed_address, positions = get_right_edge_parsed_address()
        fill_variants_parsed_address(variants_parsed_address, positions, type_street_left_edge=False)
        
        return all_variants_parsed_address
    
    def _split_by_first_number(self):
        #no type street, no house abbrevations
        variants_parsed_address = []
        address = self.address
        
        pattern_split_by_house_number = regex.compile(r'''(?<=^(.*)[-,\.\s]+)
                                            ([0-9]+)
                                            (?=[-,\.\s]*.*$)
                                            ''', regex.VERBOSE)
        
        parsed_address = pattern_split_by_house_number.search(address)
        if parsed_address is None:
            return variants_parsed_address
        
        house_number = int(parsed_address[2])
        if house_number<=0 or house_number>500:
            return variants_parsed_address
        
        street_name = parsed_address[1]
        comma = street_name.rfind(',')
        if comma > 0:
            street_name = street_name[comma+1:]
        
        if len(street_name) < self.moscow_caracteristic.MIN_STREET_LENGHT:
            return variants_parsed_address
        
        street_types = self.moscow_caracteristic.get_types_streets(street_name)
        for street_type in street_types:            
            variants_parsed_address.append(PartsAddress(street_type, street_name, house_number))
                
        return variants_parsed_address
    
    def _split_address(self):
        variants = []
        variants.extend(self._split_address_by_house())
        variants.extend(self._split_address_by_street_without_house())
        if not variants:
            variants.extend(self._split_by_first_number())
            
        return variants
        
    def _remove_modifier_street(self, street, street_type):
        modifiers = self.moscow_caracteristic.modifiers_streets
        femine = set([a[0] for a in modifiers.values()])
        masculine = set([a[1] for a in modifiers.values()])
        
        fulls_str = '|'.join(femine | masculine)
        pattern_full = regex.compile(r'''(''' + fulls_str + '''){1}
                                    ''', regex.VERBOSE)
        
        founded_modifier = pattern_full.search(street)
        if founded_modifier is not None:
            modifier = founded_modifier[1]
            street = street.replace(' ' + modifier, '')
            street = street.replace(modifier + ' ', '')
            return modifier, street
            
            
        shorts = set(modifiers.keys())
        shorts_str = '|'.join(shorts)
        pattern = regex.compile(r'''(''' + shorts_str + '''){1}
                                (?=([\. ]{1}))
                                ''', regex.VERBOSE)
        
        founded_modifier = pattern.search(street)
        if founded_modifier is None:
            pattern = regex.compile(r'''(''' + shorts_str + '''){1}
                                    (?=$)
                                    ''', regex.VERBOSE)
            founded_modifier = pattern.search(street)
            if founded_modifier is None:
                return '', street
            
            first_ending = ''
        else:
            first_ending = founded_modifier[2]
        
        modifier = founded_modifier[1]    
        street = street.replace(modifier + first_ending, '')
        street = street.strip()
        
        if modifier in modifiers:
            full_names = modifiers[modifier]
            
            if street_type in ['набережная', 'аллея', 'площадь', 'улица']:
                index = 0
            else:
                index = 1
            modifier = full_names[index]
        
        return modifier, street
    
    def _transform_street(self, street, street_type, street_modifier):
        street = street.replace('-', ' ')
        
        #delete all unvalid symbols in street
        pattern_street = regex.compile(r'[ а-я]+')
        components_street = pattern_street.findall(street)
        street = ''.join(components_street)
        street = street.strip()
        
        #сокращение "пр" для проезда пересекается с "пр-кт" и "пр-т" для проспекта, поэтому иногда получаются такие "остатки"
        if street.startswith('т '):
            street_type = 'проспект' 
            street = street[3:]
            
        elif street.startswith('кт '):
            street_type = 'проспект'
            street = street[4:] 
        
        if street.find('джалил')>=0:
            street = 'мусы джалиля'
            street_modifier = ''
                 
        return street, street_type, street_modifier
            
    def split(self):
        structured_addresses = []
        
        self._positioned_after_words_Moscow_and_Zelenograd()
        self._remove_regions()
        self._replace_abbrevations()
        self._replace_nonstandard_symbols()
        street_number = self._remove_street_number()
        
        self._remove_leading_commas_and_points()
        self.address = self.address.strip()
        if len(self.address) < 7:
            return structured_addresses
        
        variants_of_spliting_address =self._split_address()
        for variant_spliting in variants_of_spliting_address:
            street_type = variant_spliting.street_type
            street = variant_spliting.street_name
            house_number = variant_spliting.house_number 
            
            street_modifier, street = self._remove_modifier_street(street, street_type)
            street, street_type, street_modifier = self._transform_street(street, street_type, street_modifier)
               
            structured_address = {}
            structured_address['street_type'] = street_type
            structured_address['street_name'] = street.strip()
            structured_address['street_modifier'] = street_number or street_modifier
            structured_address['house_number'] =  house_number
            structured_addresses.append(structured_address) 
             
        return structured_addresses
    
    
class GeoFinder:
    def __init__(self, silent=True):
        self.silent = silent
        self.all_moscow = 0
        self.unfind_moscow = 0
        
        self.geo_base = GeoBase()
        self.moscow_characteristics = MoscowCharacteristics(self.geo_base)
        
    def get_coordinates(self, fields_addresses):
        if not fields_addresses:
            return None
        
        priority_types_addresses = []
        priority_types_addresses.append('адрес нахождения')
        priority_types_addresses.append('находится по адресу')
        priority_types_addresses.append('адрес карантина')
        priority_types_addresses.append('адрес фактического проживания')
        priority_types_addresses.append('Адрес регистрации и фактического проживания')
        priority_types_addresses.append('адрес факт. проживания')
        priority_types_addresses.append('адрес факт.проживания')
        priority_types_addresses.append('адрес мест/прожив')
        priority_types_addresses.append('адрес проживания')
        priority_types_addresses.append('проживает по адресу')
        priority_types_addresses.append('адрес')
        priority_types_addresses.append('по адресу')
        priority_types_addresses.append('адрес регистрации')
        priority_types_addresses.append('адрес прописки')
        priority_types_addresses.append('')
        
        messages = []
        splitters = [_AddressSplitter(address, self.moscow_characteristics) for address in fields_addresses]
        
        for type_address in priority_types_addresses:
            coordinates = self._find_coordinates_for_type_address(type_address, fields_addresses, splitters, messages)
            if coordinates is None:
                return None
            
            elif not coordinates:
                continue
            
            self.all_moscow += 1
            return coordinates
                
        if not self.silent:
            self.all_moscow += 1
            self.unfind_moscow += 1
            statistic = '\t' + str(self.unfind_moscow) + ' / ' + str(self.all_moscow-self.unfind_moscow) + ' / ' + str(self.all_moscow)
            
            for message in messages:
                print(message + statistic)
    
    def _find_coordinates_for_type_address(self, type_address, fields_addresses, splitters, messages):
        for splitter, field_addresses in zip(splitters, fields_addresses):
            if field_addresses.find('Грузинская д.39 кв.132')>=0:
                print('')
                
            if not splitter.positioned_by_type_address(type_address):
                continue
            
            if not splitter.this_is_Moscow():
                return None
            
            structured_addresses = splitter.split()
            for structured_address in structured_addresses:
                coordinates = self.geo_base.find_coordinates(structured_address)
                if coordinates is None:
                    messages.append(field_addresses + '\t' + str(structured_address))
                else:
                    return coordinates
        return False
                
