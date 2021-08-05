from .SimilarColumns import SimilarColumns
from . import Functions


class _Column:
    POSITION = 0
   
    def __init__(self, name, col_type):
        self.name = name
        self.col_type = col_type
        self.position = _Column.POSITION
        _Column.POSITION += 1
    
    def get_name(self):
        return self.name
    
    def get_type(self):
        return self.col_type
        
    def get_position(self):
        return self.position
    
Source =                        _Column('Source', 'U6')
MiddleNameEnding =              _Column('MiddleNameEnding', 'U3')
DateCreating =                  _Column('DateCreating', 'U10')
Birthday =                      _Column('Birthday', 'U10')
AddressOfResidence =            _Column('AddressOfResidence', 'U200')
Region =                        _Column('Region', 'U18')
DateAnalysis =                  _Column('DateAnalysis', 'U10')
CommentOfPhoneTalking =         _Column('CommentOfPhoneTalking', 'U100')
AddressArrival =                _Column('AddressArrival', 'U200')
NotFoundAtHome =                _Column('NotFoundAtHome', 'U3')
SeverityOfDiseaseAmbulance =    _Column('SeverityOfDiseaseAmbulance', 'U23')
DecisionOfAmbulance =           _Column('DecisionOfAmbulance', 'U22')
StatusDecisionOfHospitalization=_Column('StatusDecisionOfHospitalization', 'U13')
GroupOfRiskHospitalization =    _Column('GroupOfRiskHospitalization', 'U27')
StatusDecisionOfObservation =   _Column('StatusDecisionOfObservation', 'U13')
GroupOfRiskObservation =        _Column('GroupOfRiskObservation', 'U27')
CommentOfAmbulance =            _Column('CommentOfAmbulance', 'U100')
DateArrivalAmbulance =          _Column('DateArrivalAmbulance', 'U10')
DateAdmissionToHospital =       _Column('DateAdmissionToHospital', 'U10')
NameOfHospital =                _Column('NameOfHospital', 'U100')
SeverityOfDiseaseToHospital =   _Column('SeverityOfDiseaseToHospital', 'U23')
DIC =                           _Column('DIC', 'U3')
MV =                            _Column('MV', 'U3')
TransferDate =                  _Column('TransferDate', 'U10')
InitialHospital =               _Column('InitialHospital', 'U54')
TestInformation =               _Column('TestInformation', 'U23')
TypeOfPneumonia =               _Column('TypeOfPneumonia', 'U14')
ResultOfHospitalization =       _Column('ResultOfHospitalization', 'U45')
DateDepartureFromHospital =     _Column('DateDepartureFromHospital', 'U10')
CommentOfHospitalization =      _Column('CommentOfHospitalization', 'U100')
DateTreatmentInHospital =       _Column('DateTreatmentInHospital', 'U10')
AntiviralTreatment =            _Column('AntiviralTreatment', 'U3')
ListOfMedicines =               _Column('ListOfMedicines', 'U200')
ECMO =                          _Column('ECMO', 'U3')
SaturationLevel =               _Column('SaturationLevel', 'U2')
SeverityOfDiseaseInHospital =   _Column('SeverityOfDiseaseInHospital', 'U23')
DateAdmissionToObservation =    _Column('DateAdmissionToObservation', 'U10')
ResultOfObservation =           _Column('ResultOfObservation', 'U45')
DateDepartureFromObservation =  _Column('DateDepartureFromObservation', 'U10')
DateAdmissionToQuarantine =     _Column('DateAdmissionToQuarantine', 'U10')
SeverityOfDiseaseQuarantine =   _Column('SeverityOfDiseaseQuarantine', 'U23')
GroupOfRiskQuarantine =         _Column('GroupOfRiskQuarantine', 'U27')
ResultOfQuarantine =            _Column('ResultOfQuarantine', 'U45')
DateDepartureFromQuarantine =   _Column('DateDepartureFromQuarantine', 'U10')
CommentOfQuarantine =           _Column('CommentOfQuarantine', 'U100')
DateAdmissionToAftercare =      _Column('DateAdmissionToAftercare', 'U10')
ResultOfAftercare =             _Column('ResultOfAftercare', 'U45')
DateDepartureFromAftercare =    _Column('DateDepartureFromAftercare', 'U10')
CommentOfAftercare =            _Column('CommentOfAftercare', 'U100')
SeverityOfDiseaseCT =           _Column('SeverityOfDiseaseCT', 'U23')
GroupOfRiskCT =                 _Column('GroupOfRiskCT', 'U27')
ResultOfCT =                    _Column('ResultOfCT', 'U45')
DidNotTravel =                  _Column('DidNotTravel', 'U3')
Country =                       _Column('Country', 'U34')
Repeat =                        _Column('Repeat', 'U20')


all_group_of_risk = SimilarColumns('AllGroupOfRisk', [GroupOfRiskHospitalization, GroupOfRiskObservation, GroupOfRiskQuarantine, GroupOfRiskCT])
all_severity_of_disease = SimilarColumns('AllSeverityOfDisease', [SeverityOfDiseaseAmbulance, SeverityOfDiseaseToHospital, SeverityOfDiseaseInHospital, SeverityOfDiseaseQuarantine, SeverityOfDiseaseCT])
all_result = SimilarColumns('AllResult', [ResultOfHospitalization, ResultOfObservation, ResultOfQuarantine, ResultOfAftercare, ResultOfCT])
all_comment = SimilarColumns('AllComment', [CommentOfAmbulance, CommentOfPhoneTalking, CommentOfHospitalization, CommentOfQuarantine, CommentOfAftercare])

all_names_hospitalizataion = Functions.get_columns_by_part_name('Hospital', converted=False)
all_names_hospitalizataion.extend(['DIC', 'MV', 'TransferDate', 'TestInformation', 'TypeOfPneumonia', 'AntiviralTreatment', 'ListOfMedicines', 'ECMO', 'SaturationLevel'])
        
all_names_obervation = Functions.get_columns_by_part_name('Observation', converted=False)
all_names_quarantine = Functions.get_columns_by_part_name('Quarantine', converted=False)
all_names_aftercare = Functions.get_columns_by_part_name('Aftercare', converted=False)

