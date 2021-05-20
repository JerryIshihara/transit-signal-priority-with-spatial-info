# cd/d C:\Python27 & for /l %x in (1176291, 1, 1180590) do (
#     python "C:\Users\Public\Documents\ShalabyGroup\Aimsun Controller\RunSeveralReplications.py" -aconsolePath "C:\Program Files\Aimsun\Aimsun Next 8.3\aconsole.exe" -modelPath "C:\Users\Public\Documents\ShalabyGroup\TSP-Louis\finchTSPs_3 intx_west_Subnetwork 1171379.ang" -targets %x
#     )

# & "C:\\Program Files\\Aimsun\\Aimsun Next 8.3\\aconsole.exe" 
# -v -log -project "C:\\Users\\siwei\\Documents\\Developer\\aimsun\\finchTSPs_3 intx_west_Subnetwork 1171379.ang" -cmd execute -target 1180681



import socket
from subprocess import Popen, PIPE
import time
import numpy as np
import random

HOST = 'localhost'
PORT = 23000 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(10)
print('[Info] Aimsun Manager Ready. Waiting For Aimsun Instance...')

process = Popen(['"aconsole.exe"', '-v', '-log', 
                    '-project', 'C:\\Users\\siwei\\Documents\\Developer\\aimsun\\finchTSPs_3 intx_west_Subnetwork 1171379.ang',
                    '-cmd', 'execute', '-target', '1180681'], executable="C:\\Program Files\\Aimsun\\Aimsun Next 8.3\\aconsole.exe")
# process = Popen(['"aconsole.exe"'], executable="C:\\Program Files\\Aimsun\\Aimsun Next 8.3\\aconsole.exe")
# def start_aimsun_instance(self, s):
#     process = Popen(['python', 'aimsun.py'])
#     print("Waiting for aimsun instance to connect...")
#     conn, addr = s.accept()
#     print('[Info] Connected by', addr)
#     conn.send(b'SYN')
#     data = conn.recv(1024).decode("utf-8")
#     if data != "SYN":
#         print("[ERROR] Handshake Failed.")
#         return False, -1 
#     else:
#         print("[Info] Aimsun Instance connected.")
#         return True, conn
# process = Popen(['python', 'aimsun.py'])
conn, addr = s.accept()
print('Connected by', addr)

repID = 1180681
sync_message = 'SYN' + str(repID)
conn.send(bytes(sync_message, 'utf-8'))
data = conn.recv(1024).decode("utf-8")
if data != "SYN":
    print("[ERROR] Handshake Failed.")
else:
    print("[Info] Aimsun Instance connected.")
    


while True:
    conn.send(b'GET_STATE')
    data = conn.recv(1024).decode("utf-8")
    if(data == 'FIN'):
        print("FIN Received")
        break
    elif(data[:10] != 'DATA_READY'):
        print("ERROR")
        break
    else:
        time = data[10:]
        feature = np.load('realtime_state.npy')
        print(feature.shape)
        print("Time: " + time)

        # apply action here
        rand = random.randint(0, 100)
        if rand < 90:
            conn.send(b'WRITE_ACTION:EXTEND')
        else:
            conn.send(b'WRITE_ACTION:NOTHING')
            
s.close()
print('END')