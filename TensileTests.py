# -*- coding: utf-8 -*-
"""
Created on Wed May 15 07:05:55 2024

@author: seede
"""

import numpy as np
import csv
from matplotlib import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#Initialise
gf = -0.00980665


class TestSample:
    def __init__(self, thickness1, thickness2, thickness3):
        self.thickness1 = thickness1
        self.thickness2 = thickness2
        self.thickness3 = thickness3
        self.mean_thickness = np.mean([self.thickness1, self.thickness2, self.thickness3])
        self.gauge_length = -30  # mm
        self.width = 10 #mm
        self.length = 50 #mm
        self.area = self.mean_thickness *  self.width #mm^2
        self.YoungsModulus = float()
    
    def display_thicknesses(self):
        print("Thickness 1:", self.thickness1)
        print("Thickness 2:", self.thickness2)
        print("Thickness 3:", self.thickness3)
        
        
#Function which takes the test_sample object i.e. AC_107um_1, the raw Mach-1 data file, and the data series name for the graph
def TENSILE_TESTS(test_sample, file_path, data_frame_name):
    
    def load_data(file_path, start_line=24):
    # Read data into a DataFrame
        data = pd.read_csv(file_path, delimiter='\t', skiprows=start_line - 1, header=None)

    # Assign headers to the columns
        data.columns = ['Time', 'Position', 'Fz,gf']
    
    # Clean the data 
    # Remove rows where 'Position' is equal to 0
        data = data[data['Position'] != 0]
        data = data.iloc[5:]
    
    # Convert 'Fz,gf' column to numeric type, handle non-numeric values by replacing them with NaN
        data['Fz,gf'] = pd.to_numeric(data['Fz,gf'], errors='coerce')
    
    # Remove rows with NaN values in the 'Fz,gf' column
        data = data.dropna(subset=['Fz,gf'])
    
    # Calculate new column values
        data['Fz, N'] = data['Fz,gf'] * gf  # Calculates force in Newtons

        return data  
    

    def calc_stress_strain(test_sample, loaded_data):
        # Calculate the strain: change in position / original gauge_length
        strain = loaded_data['Position'] / test_sample.gauge_length

        # Calculate the stress as the Force [N] / (sample width * sample mean thickness) [mm^2]
        stress = loaded_data['Fz, N'] / (test_sample.width * test_sample.mean_thickness)

        # Create a DataFrame with strain and stress columns
        stress_strain = pd.DataFrame({'Strain': strain, 'Stress': stress})

        # Convert the 'Stress' column to floats
        stress_strain['Stress'] = stress_strain['Stress'].astype(float)

        return stress_strain  # Return the DataFrame directly without wrapping it in a list

    def plot_linear_fit_and_modulus(data_frame, data_frame_name, start_strain=0.05, end_strain=0.10):
        # Filter the data for strains between start_strain (5%) and end_strain (10%)
        subset_between_start_and_end = data_frame[(data_frame['Strain'] >= start_strain) & (data_frame['Strain'] <= end_strain)]

        # Fit a linear regression line to the subset
        linear_coefficients = np.polyfit(subset_between_start_and_end['Strain'], subset_between_start_and_end['Stress'], 1)
        linear_model = np.poly1d(linear_coefficients)

        # Plot stress-strain curve
        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")

        # Plot stress-strain data points
        plt.scatter(data_frame['Strain'], data_frame['Stress'], color='blue', label=data_frame_name + ' Data Points')

        # Plot linear regression line for strains between start_strain and end_strain
        strain_range = np.linspace(start_strain, end_strain, 100)
        plt.plot(strain_range, linear_model(strain_range), color='red', label='Linear Fit between {:.2f} and {:.2f} Strain'.format(start_strain, end_strain))

        # Set labels and title
        plt.xlabel('Strain')
        plt.ylabel('Stress')
        plt.title('Stress-Strain Curve')

        # Add legend
        plt.legend()

        # Show plot
        plt.show()

        # Calculate and return Young's Modulus (gradient of the linear fit line)
        return linear_coefficients[0]

    # Load the data
    loaded_data = load_data(file_path)

    # Calculate stress-strain
    stress_strain_data = calc_stress_strain(test_sample, loaded_data)
    
    #print the Young's modulus
    print("Young's Modulus")
    # Plot stress-strain curve and return Young's Modulus
    return plot_linear_fit_and_modulus(stress_strain_data, data_frame_name)


#LOAD TEST SAMPLES with the thicnkesses
AC_170um_1 = TestSample(0.150, 0.175, 0.193)
AC_170um_2 = TestSample(0.185, 0.173, 0.154)
AC_170um_3 = TestSample(0.154, 0.194, 0.221)
AC_170um_4 = TestSample(0.169, 0.196, 0.220) 
AC_170um_5 = TestSample(0.156, 0.176, 0.202)
AC_170um_6 = TestSample(0.193, 0.216, 0.191)

P355D_300um_1 = TestSample(0.304, 0.134, 0.287)
P355D_300um_2 = TestSample(0.366, 0.364, 0.316)
P355D_300um_3 = TestSample(0.306, 0.312, 0.314)
P355D_300um_4 = TestSample(0.385, 0.303, 0.321)
P355D_300um_5 = TestSample(0.362, 0.330, 0.377)


#Function TENSILE_TESTS(1, '2', '3') takes the Test Sample defined above, i.e. AC_170um_1
#then 2 is the 'filepath' with raw data from Mach-1 machine, then '3' is a string for what 
#the data series should be called in the graph
#The Young's modulus is stored in the Test Sample object, find with AC_170um_1.YoungsModulus for example

AC_170um_1.YoungsModulus = TENSILE_TESTS(AC_170um_1, 'C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/AC4095A 30Apr24/test1.txt', 'AC-170um-1')
AC_170um_2.YoungsModulus = TENSILE_TESTS(AC_170um_2, 'C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/AC4095A 30Apr24/test2.txt', 'AC-170um-2')
AC_170um_3.YoungsModulus = TENSILE_TESTS(AC_170um_3, 'C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/AC4095A 30Apr24/test3.txt', 'AC-170um-3')
AC_170um_4.YoungsModulus = TENSILE_TESTS(AC_170um_4, 'C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/AC4095A 30Apr24/test4.txt', 'AC-170um-4')
AC_170um_5.YoungsModulus = TENSILE_TESTS(AC_170um_5, 'C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/AC4095A 30Apr24/test5.txt', 'AC-170um-5')
AC_170um_6.YoungsModulus = TENSILE_TESTS(AC_170um_6, 'C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/AC4095A 30Apr24/test6.txt', 'AC-170um-6')


P355D_300um_1.YoungsModulus = TENSILE_TESTS(P355D_300um_1, 'C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/P355D 1May24/test2_1.txt', 'P355D_300um-1')
P355D_300um_2.YoungsModulus = TENSILE_TESTS(P355D_300um_2, "C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/P355D 1May24/test2_2.txt", 'P355D_300um-2')
P355D_300um_3.YoungsModulus = TENSILE_TESTS(P355D_300um_3, "C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/P355D 1May24/test2_3.txt", 'P355D_300um-3')
P355D_300um_4.YoungsModulus = TENSILE_TESTS(P355D_300um_4, "C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/P355D 1May24/test2_4.txt", 'P355D_300um-4')
P355D_300um_5.YoungsModulus = TENSILE_TESTS(P355D_300um_5, "C:/Users/seede/OneDrive/Documents/PhD/Tensile Tests/Ellie/P355D 1May24/test2_5.txt", 'P355D_300um-5')








