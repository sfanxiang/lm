import matplotlib.pyplot as plt
import numpy as np

tokens = np.arange(10,101,5)

#rust_data = [ 0.923, 1.669, 2.21, 2.853, 3.394, 3.601, 4.075, 4.399, 4.608]

#python_data = [ 1.264, 1.820, 2.299, 2.748, 3.771, 3.736, 4.427, 4.903, 5.561]


python_data = [ 0.5705447196960449, 0.8337748050689697, 1.0924952030181885, 1.3839828968048096, 1.8377423286437988, 2.0831985473632812, 2.2969391345977783, 2.4464986324310303, 2.735520124435425, 3.086606025695801, 3.362382411956787, 3.5417823791503906, 4.142303228378296, 4.369992256164551, 4.9922096729278564, 4.700778007507324, 4.8550779819488525, 5.014612674713135, 6.936026334762573]

rust_data = [ 0.496, 0.807, 1.267, 1.422, 1.609, 1.711, 4.546, 2.169, 2.559, 2.685, 2.917, 3.425, 3.583, 3.678, 5.148, 4.497, 4.314, 4.535, 4.736]

#we do a little manual regression 
#rust_data[6] = 1.952 
#rust_data[14] = 4.323

plt.plot(tokens, rust_data, label = 'Rust' , color = 'red')
plt.plot(tokens, python_data, label = 'Python' , color = 'blue')
plt.xlabel("Number of Tokens")
plt.ylabel("Time [seconds]")
plt.title("Speed of Generating Tokens in Rust vs Python ")
plt.legend()
plt.savefig("TimePlot.pdf")
plt.show()