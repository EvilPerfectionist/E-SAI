import numpy as np

event_path = "Example_data/Raw/Event/0001.npy"
data = np.load(event_path,allow_pickle=True).item()
print(data)
eventData = data.get('events')

# x = eventData[:,0]
# y = eventData[:,1]
# t = eventData[:,2]
# p = eventData[:,3]

# with open('events.txt', 'w') as f:
#     for i in range(len(x)):
#         line = [str(t[i]), ' ', str(int(x[i])), ' ', str(int(y[i])), ' ', str(int(p[i])), '\n']
#         f.writelines(line)