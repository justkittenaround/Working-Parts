import numpy as np
import matplotlib.pyplot as plt

def histogram(data):
    '''This function takes in a vector called data and 
        plots a histogram using the values.'''
    
    fig = plt.figure(figsize=(7, 5))
    plot1 = fig.add_subplot(111)
    plot1.set_xlabel('Value')
    plot1.set_ylabel('Frequency')
    plt.hist(data, bins=100)
    plt.show()