import pandas as pd
import numpy as np

# Prepares pre-pilot data
def prepare_data_pre():
    # df = pd.read_csv('GreenSoul_All_Data_v01.csv')
    df = pd.read_csv('old_dataset_v3.csv', sep = ';')
    df[df == '?'] = np.nan
    
    
#    attr_dict = {
#        'Age': 'age',
#        'Genre': 'gender',
#        'Children': 'children',
#        'Education': 'education',
#        'Country': 'country',
#        'City': 'city',
#        'Employment': 'employment',
#        'Position': 'position',
#        'Work_culture': 'work_culture',
#        'Floor': 'floor',
#        'Sharing': 'sharing',
#        'Work_activity': 'work_activity',
#        'Thermal': 'thermal',
#        'Profile': 'profile',
#        # 'Top_energy',
#        'Organisation_energy': 'org_energy',
#        'Barriers': 'barriers',
#        'Intentions': 'intentions',
#        'Confidence': 'confidence',
#        'How_often': 'how_often',
#        'Temperature_winter': 'temperature_winter',
#        'Temperature_summer': 'temperature_summer',
#        'Lighting_workplace': 'lighting_workplace',
#        'Consensus': 'consensus',
#        'Stairs_elevator': 'stairs_elevator',
#        'printing_avoid': 'printing_avoid',
#        'printing_delay': 'printing_delay',
#        'Efficient_mode': 'efficient_mode',
#        'Switchoff_stop': 'switchoff_stop',
#        'Switchoff_breaks': 'switchoff_breaks',
#        'Sacrifice_winter': 'sacrifice_winter',
#        'Wear_dress': 'wear_dress',
#        'Wear_casual': 'wear_casual',
#        'Open_windows': 'open_windows',
#        'Influenceness': 'influenceness',
#        'Susceptibility': 'susceptibility',
#        'Initiative_join': 'initiative_join',
#        'Frequency': 'frequency',
#        'Response_signs': 'response_signs',
#        
#        'Social_Recognition': 'social_recognition',
#        'Self_monitoring': 'self_monitoring',
#        'Suggestions': 'suggestion',
#        'Appraisal': 'praise',
#        'Peer_Pressure': 'similarity',
#        'Rewards': 'conditioning',
#        'Convenience_Flexibility': 'physical_attractiveness',
#        'Trust_Validity': 'trust_validity',
#        'Self_assessment': 'self_assessment'
#    }
    
    attr_dict = {
        'Age': 'age',
        'Genre': 'gender',
        'Education': 'education',
        'Country': 'country',
        'City': 'city',
        'Employment': 'employment',
        'Position': 'position',
        'Work_culture': 'work_culture',
        'Floor': 'floor',
        'Sharing': 'sharing',
        'Work_activity': 'work_activity',
        'Profile_PST': 'profile',
        'Intentions': 'intentions',
        'Confidence': 'confidence',
        'Organisation_energy': 'organisation_energy',
        'Barriers': 'barriers',
        'Consensus': 'consensus',
        'Influenceness': 'influenceness',
        'Susceptibility': 'susceptibility',
        'Initiative_join': 'initiative_join',
        'Frequency': 'frequency',
        'Response_signs': 'response_signs',
        'Thermal': 'thermal',
        'How_often': 'how_often',
        'Temperature_winter': 'temperature_winter',
        'Temperature_summer': 'temperature_summer',
        'Lighting_workplace': 'lighting_workplace',
        'Stairs_elevator': 'stairs_elevator',
        'Printing_avoid': 'printing_avoid',
        'Printing_delay': 'printing_delay',
        'Efficient_mode': 'efficient_mode',
        'Switchoff_stop': 'switchoff_stop',
        'Switchoff_breaks': 'switchoff_breaks',
        'Sacrifice': 'sacrifice',
        'Wear_dress': 'wear_dress',
        'Wear_casual': 'wear_casual',
        'Open_windows': 'open_windows',
        
        'Authority': 'authority',
        'Cause_effect': 'cause_effect',
        'Conditioning': 'conditioning',
        'Cooperation & Liking ': 'cooperation_liking ',
        'Tailoring & Personalisation ': 'tailoring_personalisation ',
        'Physical Attractiveness': 'physical_attractiveness',
        'Praise': 'praise',
        'Verifiability': 'verifiability',
        'Reciprocity': 'reciprocity',
        'Reduction ': 'reduction ',
        'Self_monitoring': 'self_monitoring',
        'Similarity': 'similarity',
        'Social_proof': 'social_proof',
        'Social_recognition': 'social_recognition',
        'Suggestion': 'suggestion'
    }
    
    attrs = list(attr_dict.keys())
    
    df = df[attrs]
    df.rename(columns = attr_dict, inplace = True)
    
    return df

def prepare_data_post():
    # df = pd.read_csv('Post-pilot_ALL_Raking-CLEANED.csv', sep = ';')
    df = pd.read_csv('Post-pilot_ALL_Raking-CLEANED - Ruben NDPM.csv', sep = ',')
    
    attr_dict = {
        'Age': 'age',
        'Gender': 'gender',
        'Education': 'education',
        'City': 'city',
        'Position': 'position',
        'Work_culture': 'work_culture',
        'Profile_PST': 'profile',
        'Barriers': 'barriers',
        'Intentions': 'intentions',
        'Confidence': 'confidence',
        'Initiative_join': 'initiative_join',
        'Frequency': 'frequency',
        
#        'Social_recognition': 'social_recognition',
#        'Self_monitoring': 'self_monitoring',
#        'Suggestion': 'suggestion',
#        'Similarity': 'similarity',
#        'Conditioning': 'conditioning',
#        'Cause_effect': 'cause_effect',
#        'Physical_attractiveness': 'physical_attractiveness',
#        'Reciprocity': 'reciprocity',
#        'Authority': 'authority',
#        'Social_proof': 'social_proof'
        
        'v2': 'social_recognition',
        'v11': 'self_monitoring',
        'v19': 'suggestion',
        'v20': 'similarity',
        'v6': 'conditioning',
        'v15': 'cause_effect',
        'v5': 'physical_attractiveness',
        'v7': 'reciprocity',
        # 'Authority': 'authority',
        'v10': 'social_proof'
    }
    
    attrs = list(attr_dict.keys())
    
    df = df[attrs]
    df.rename(columns = attr_dict, inplace = True)
    
    return df

def get_predictors_targets():
    # full list of predictor variables (available in pre-pilot)
    predictor_vars_full = [
        'age',
        'gender',
        'education',
        'country',
        'city',
        'employment',
        'position',
        'work_culture',
        'floor',
        'sharing',
        'work_activity',
        'profile',
        'intentions',
        'confidence',
        'organisation_energy',
        'barriers',
        'consensus',
        'influenceness',
        'susceptibility',
        'initiative_join',
        'frequency',
        'response_signs',
        'thermal',
        'how_often',
        'temperature_winter',
        'temperature_summer',
        'lighting_workplace',
        'stairs_elevator',
        'printing_avoid',
        'printing_delay',
        'efficient_mode',
        'switchoff_stop',
        'switchoff_breaks',
        'sacrifice',
        'wear_dress',
        'wear_casual',
        'open_windows'
    ]
    
    # limited list of predictor variables (available in post-pilot)
    predictor_vars_ltd = [
        'age',
        'gender',
        'education',
        'city',
        'position',
        'work_culture',
        'profile',
        'barriers',
        'intentions',
        'confidence',
        'initiative_join',
        'frequency'
    ]
    
#    predictor_vars_ltd = [
#        'work_culture',
#        'frequency',
#        'initiative_join',
#        'confidence',
#        'city',
#        'gender',
#        'profile',
#        'age'
#    ]
    
    # list of target variables
    target_vars = [
        'social_recognition',
        'self_monitoring',
        'suggestion',
        'similarity',
        'conditioning',
        'cause_effect',
        'physical_attractiveness',
        'reciprocity',
        # 'authority',
        'social_proof'
    ]
    
    return predictor_vars_full, predictor_vars_ltd, target_vars
