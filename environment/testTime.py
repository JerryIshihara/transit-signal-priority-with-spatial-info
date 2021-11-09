from AAPI import *
import math
import numpy as np
import socket
from thread import start_new_thread
from threading import Lock

GRID_SIZE = 3
SAFETY_GRID = 10
IN_STOP_RANGE = 0.5 # range that decide whether the bus is in the stop or not
TARGET_HEADWAY = 290

STATE_PATH = "C:\\Users\\siwei\\Documents\\Developer\\transit-signal-priority-with-spatial-info\\environment\\realtime_state.npy"

globalLock = Lock()
threadCommMsg = "" # global variable for comm between main program and threads
g_logFile = None

# [warning] only support one EB and then one WB road
class global_data:
    def __init__(self, _road_seg_arr, _configuration_list, _intersection_list):
        self.road_seg_arr = _road_seg_arr
        self.intersection_list = _intersection_list
        
        intersection_id = -1
        interested_phase = []
        for intersection in _intersection_list:
            if (intersection_id == -1):
                intersection_id = intersection['intersection']
            else:
                assert(intersection_id == intersection['intersection'])
            
            for phase in intersection["phase_of_interest"]:
                if (phase not in interested_phase):
                    interested_phase.append(phase)
        self.m_unique_intersection = intersection_id
        self.m_interested_phases = interested_phase

        # construct segment mapping from segment id to index in road_seg_arr
        self.segment_mapping = {}
        road_seg_idx = 0
        _status_grid_np_length = 0
        for road_reg in _road_seg_arr:
            for road_sec in road_reg.section_list:
                self.segment_mapping[road_sec] = road_seg_idx
            road_seg_idx += 1
            _status_grid_np_length += 4

        total_width = 0
        total_length = 0
        # construct feature grid and set road_seg direction
        for idx in range(len(_configuration_list)):
            if _configuration_list[idx][0]:
                total_width = max(_road_seg_arr[idx].lane_num)
                total_length = int(np.ceil(sum(_road_seg_arr[idx].section_length)/ GRID_SIZE))
            elif _configuration_list[idx][1]:
                total_width += max(_road_seg_arr[idx].lane_num)
                total_length += int(np.ceil(sum(_road_seg_arr[idx].section_length)/ GRID_SIZE))
        total_length += SAFETY_GRID

        self.feature_grid_np = np.zeros([total_width, total_length, 0])
        self.status_grid_np = np.zeros([0, _status_grid_np_length])
        self.head_way = [-1 for i in range (len(_road_seg_arr))]

        # reward related
        self.requestReward = False
        self.lastTimestepTime = -1
        self.lastTimestepHeadway = [-1, -1] # valid time step range
        self.lastTimestepBusNum = [-1, -1]
        self.busOutNotCapturedHeadway = [-1, -1]
        self.closingReward = [-1, -1]

    def hash_position(self, num_of_seg, distance_to_intersection, lane):
        x, y = self.road_seg_arr[num_of_seg].hash_position(distance_to_intersection, lane)
        if num_of_seg == 0:
            x = math.ceil(self.road_seg_arr[num_of_seg].total_len / GRID_SIZE) + SAFETY_GRID - x
        elif num_of_seg == 1:
            x = x + math.ceil(self.road_seg_arr[0].total_len / GRID_SIZE) + SAFETY_GRID
            y = max(self.road_seg_arr[0].lane_num) + (max(self.road_seg_arr[num_of_seg].lane_num) - 1 - y)
        return int(x), int(y)

    def hash_car_coor(self, time, car):
        # print("global hash_car_coor")
        num_of_seg = self.segment_mapping[car.idSection]
        x, y = self.road_seg_arr[num_of_seg].hash_car_coor(time, car)
        # print(str(x))
        # print(str(y))
        if num_of_seg == 0:
            x = math.ceil(self.road_seg_arr[num_of_seg].total_len / GRID_SIZE) + SAFETY_GRID - x
        elif num_of_seg == 1:
            x = x + math.ceil(self.road_seg_arr[0].total_len / GRID_SIZE) + SAFETY_GRID
            y = max(self.road_seg_arr[0].lane_num) + (max(self.road_seg_arr[num_of_seg].lane_num) - 1 - y)

        # print("global hash_car_coor END")
        return int(x), int(y)

    def get_segment_list(self):
        ret_list = []
        for segment in self.road_seg_arr:
            for section in segment.section_list:
                ret_list.append(section)

        return ret_list

    def encode_car_list(self, time, car_list):
        # AKIPrintString(str(dist2end))
        # 0          || left_0 --- right_(dist2end[0] / GRID_SIZE) - 1            /
        # ...        ||                      ...                                 / ------ Car coming this way  <---<---
        # ...        ||                      ...                                 \ ------ <---<---   <---<---  <---<---
        # ROAD_WDIH_MAX - 1 || left_0 --- right_(dist2end[0] / GRID_SIZE) - 1     \
        tmp_feature_grid = np.zeros([self.feature_grid_np.shape[0], self.feature_grid_np.shape[1]])

        for car in car_list:
            # print("encode check0")
            x, y = self.hash_car_coor(time, car)
            # print(str((x, y)))
            tmp_feature_grid[y][x] = car.type
        # print("encode check2")
        self.feature_grid_np = np.dstack((self.feature_grid_np, tmp_feature_grid))

    def update_status_grid(self, time):
        new_status = np.array([])
        # AKIPrintString(str(self.feature_grid_np.shape))
        # for segment_idx in range(len(self.road_seg_arr)):
        #     # calculate dwell time
        #     if self.feature_grid_np[:,:,self.feature_grid_np.shape[2] - 1][0][2] == 3:  # bus in the station
        #         if (self.road_seg_arr[segment_idx].last_stop_time != 0):
        #             print(self.road_seg_arr[segment_idx].last_stop_time)
        #         self.road_seg_arr[segment_idx].last_stop_time = 0
        #         self.road_seg_arr[segment_idx].dwell_time += 1
        #     else:
        #         self.road_seg_arr[segment_idx].last_stop_time += 1
        #         self.road_seg_arr[segment_idx].dwell_time = 0

        #     # calculate one layer of status_grid_np
        #     # phase = get_phase_number(time, self.intersection_list[segment_idx]['phase_duration'])
        #     time_to_green = time_to_phase_end(
        #         time, self.intersection_list[segment_idx]['phase_duration'], self.intersection_list[segment_idx]['phase_of_interest'])

        #     new_status = np.concatenate([new_status, np.array([time_to_green, self.road_seg_arr[segment_idx].dwell_time])])
        # self.status_grid_np = np.vstack([self.status_grid_np, new_status])

    def post_update(self, time):
        new_status = np.array([])
        for road_seg_idx in range(len(self.road_seg_arr)):
            # update head_way here
            headway, since_last_bus_stop, since_last_bus_out, phase_time, dwell_time = self.road_seg_arr[road_seg_idx].post_update(time)
            if headway != -1:
                self.head_way[road_seg_idx] = headway

            new_status = np.concatenate([new_status, np.array([since_last_bus_stop, since_last_bus_out, phase_time, dwell_time])])
        self.status_grid_np = np.vstack([self.status_grid_np, new_status])

class road_seg:
    def __init__(self, _intersection_info, _section_list, _section_length, _lane_num, _lane_offset, _bus_stop_distance_to_end, _is_has_bus_stop):
        self.intersection_info        = _intersection_info
        self.section_list             = _section_list
        self.section_length           = _section_length
        self.lane_num                 = _lane_num
        self.lane_offset              = _lane_offset
        
        self.last_stop_time           = -1
        self.dwell_time               = 0
        self.last_bus_out             = -1
        self.prev_bus_num             = -1
        self.prev_closest_to_exit_bus = -1 # the bus ID that is the closest to the checkout

        self.closest_to_exit_bus      = [999999, -1] # the bus ID that is the closest to the checkout
        self.bus_num                  = 0
        self.was_bus_in_stop          = False # bus in stop status in the last time
        self.is_bus_in_stop           = False
        self.dynamic_headway          = -1

        self.total_len                = sum(_section_length)
        self.dist2end                 = [0 for i in range(len(_section_list))]
        self.bus_stop_distance_to_end = _bus_stop_distance_to_end
        self.is_has_bus_stop          = _is_has_bus_stop

        i = len(_section_list) - 1
        acc_sum = 0
        while i >= 0:
            acc_sum += _section_length[i]
            self.dist2end[i] = acc_sum
            i -= 1

    def before_hash_cars(self):
        # tmp var to find the bus closest to the bus station
        self.fastest_bus_id              = -1
        self.fastest_bus_distance_to_end = 999999
        self.closest_to_exit_bus         = [999999, -1] ## idx0: distance2End, idx1: bus ID
        self.bus_num                     = 0
        self.is_bus_in_stop              = False

    def hash_position(self, distance_to_intersection, lane):
        return int(math.floor(distance_to_intersection / GRID_SIZE)), lane

    def hash_car_coor(self, time, car):
        # print("road_seg hash_car_coor")
        num_of_sec = self.section_list.index(car.idSection)
        distance_to_intersection = (car.distance2End + self.dist2end[num_of_sec + 1]) if (
            num_of_sec < len(self.section_list) - 1) else (car.distance2End)
        
        # print(str(self.dist2end))
        # print(str(distance_to_intersection))
        if (car.type == 3):
            self.bus_num += 1
            if distance_to_intersection < self.closest_to_exit_bus[0]:
                self.closest_to_exit_bus = [distance_to_intersection, car.idVeh]
        if (car.type == 3 and distance_to_intersection > self.bus_stop_distance_to_end and distance_to_intersection < self.fastest_bus_distance_to_end):
            self.fastest_bus_id = car.idVeh
            self.fastest_bus_distance_to_end = distance_to_intersection

        if (self.is_has_bus_stop and car.type == 3 and abs(distance_to_intersection - self.bus_stop_distance_to_end) <= IN_STOP_RANGE):
            self.is_bus_in_stop = True

        # print("road_seg hash_car_coor END")
        return int(math.floor(distance_to_intersection / GRID_SIZE)), int(car.numberLane - 1 + self.lane_offset[num_of_sec])

    # def get_bus_info(self):
        # print(str(self.fastest_bus_distance_to_end))
        # print(str(self.fastest_bus_id))

    def post_update(self, time):
        self.dynamic_headway = -1
        if (self.is_bus_in_stop):
            self.last_stop_time = time

        if (self.last_bus_out != -1):
            # self.bus_num = 0
            if (self.bus_num > 0):
                self.dynamic_headway = time - self.last_bus_out

        # if (self.bus_num < self.prev_bus_num) :
        #     self.last_bus_out = time
        # print(self.closest_to_exit_bus[1], self.prev_closest_to_exit_bus)
        if (self.closest_to_exit_bus[1] != self.prev_closest_to_exit_bus and self.prev_closest_to_exit_bus != -1) :
            self.last_bus_out = time

        
        if self.last_stop_time == -1:
            since_last_bus_stop = -1
        else:
            since_last_bus_stop = time - self.last_stop_time

        if self.last_bus_out == -1:
            since_last_bus_out = -1
        else:
            since_last_bus_out = time - self.last_bus_out

        # time_to_green = time_to_phase_end(
        #         time, self.intersection_info["phase_duration"], self.intersection_info["phase_of_interest"])
        current_phase_num = ECIGetCurrentPhase(self.intersection_info["intersection"]) - 1
        current_phase_time = time - ECIGetStartingTimePhase(self.intersection_info["intersection"])

        if (self.is_bus_in_stop):
            self.dwell_time += 1
        else:
            self.dwell_time = 0 

        if (not self.is_has_bus_stop):
            since_last_bus_stop = -1
            self.dwell_time = -1
        
        self.was_bus_in_stop = self.is_bus_in_stop
        self.prev_bus_num = self.bus_num
        self.prev_closest_to_exit_bus = self.closest_to_exit_bus[1]
        # return headway, since_last_bus_stop, since_last_bus_out, time_to_green, self.dwell_time
        # headway, time since last pass bus stop and time since last bus out road segment, current phase time, and dwell_time
        return self.dynamic_headway, since_last_bus_stop, since_last_bus_out, current_phase_time, self.dwell_time

## helper utilization functions
def get_car_info(section_list, junction_list):
    car_info_list = []

    for id in section_list:
        nb = AKIVehStateGetNbVehiclesSection(id, True)
        for j in range(nb):
            infVeh = AKIVehStateGetVehicleInfSection(id, j)
            car_info_list.append(infVeh)
            # astring = "Vehicle " + str(infVeh.idVeh) + ", Section " + str(infVeh.idSection) + ", Type " + str(infVeh.type) + " , Lane " + str(
            #     infVeh.numberLane) + ", CurrentPos " + str(infVeh.CurrentPos) + ", CurrentSpeed " + str(infVeh.CurrentSpeed)
            # AKIPrintString(astring)

    # for id in junction_list:
    #     nb = AKIVehStateGetNbVehiclesJunction(id)
    #     for j in range(nb):
    #         infVeh = AKIVehStateGetVehicleInfJunction(id, j)
    #         car_info_list.append(infVeh)
    return car_info_list


def time_to_phase_end(time, phases, phase_of_interest):
    global global_data_flow
    intersection = global_data_flow.m_unique_intersection

    phase_num = ECIGetCurrentPhase(intersection) - 1
    current_phase_time = time - ECIGetStartingTimePhase(intersection)
    remaining_current_phase_time = phases[phase_num] - current_phase_time

    if phase_num < phase_of_interest:
        return remaining_current_phase_time + np.sum(phases[phase_num + 1:phase_of_interest + 1])
    elif phase_num == phase_of_interest:
        return remaining_current_phase_time
    elif phase_num != len(phases) - 1:
        return remaining_current_phase_time + np.sum(phases[phase_num + 1:len(phases)]) + np.sum(phases[0:phase_of_interest + 1])
    else:
        return remaining_current_phase_time + np.sum(phases[0:phase_of_interest + 1])

## main flow
def init_all():
    global global_data_flow
    intersection_list = [{
                            'intersection': 1171274,
                            'phase_duration': [16, 38, 7, 11, 32, 6],  # focus on 32s
                            'phase_of_interest': [1, 4]
                        }, {
                            'intersection': 1171274,
                            'phase_duration': [16, 38, 7, 11, 32, 6],  # focus on 32s
                            'phase_of_interest': [1, 4]
                        }]
    road_segs = [road_seg(intersection_list[0], [1174260, 6601], [164.316, 186.013], [2, 4], [1, 0], 8.2, True), \
                 road_seg(intersection_list[1], [6607], [155.572], [4], [0], -1, False)]
    configuration_list = [[True, False], [False, True]] # [EB, WB] doesn't support both direction for nows
    global_data_flow = global_data(road_segs, configuration_list, intersection_list)


def AAPILoad():
    return 0

def threaded_client(s):
    global globalLock
    global threadCommMsg
    global g_logFile
    data = s.recv(1024).decode("utf-8")
    if data[:3] != 'SYN':
        s.close()
        print("[ERROR] Handshake error.")
        return

    data = data.decode("utf-8")
    repID = data[3:]
    print("repID" + repID)
    g_logFile = open("Aimsun_Log_" + repID + ".csv", 'w')
    g_logFile.write(",".join(["time", "currentPhase(start 0)", "phasetime", "bus num-EB", "last stop-EB", "last checkout-EB", "headway-EB", \
                              "bus num-WB", "last stop-WB", "last checkout-WB", "headway-WB", "notify", "lastcheckoutEB", "lastcheckoutWB", "reward", "action"]) + "\n")

    s.send(b'SYN')
    while True:
        data = s.recv(1024)
        data = data.decode("utf-8")

        if data == "FIN":
            break
        elif data == "GET_STATE":
            globalLock.release()
        elif data == "ACTION_READY":
            globalLock.release()
        elif data[:13] == "WRITE_ACTION:":
            threadCommMsg = data[13:]
            globalLock.release()
    print("FIN flag received")
    s.close()

def sendmsg(msg):
    global globalSocket
    globalSocket.send(msg)

    if msg == "FIN":
        globalSocket.close()

def AAPIInit():
    global globalLock
    global globalSocket
    AKIPrintString("Start")
    init_all()

    HOST = 'localhost'
    PORT = 23000 
    globalSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    globalSocket.connect((HOST, PORT))

    globalLock.acquire()
    start_new_thread(threaded_client, (globalSocket,))
    # globalLock.acquire()
    return 0


def AAPIManage(time, timeSta, timTrans, SimStep):
    # print("[AIMSUN DEBUG] AAPIManage")
    global globalLock
    global global_data_flow

    for seg in global_data_flow.road_seg_arr:
        seg.before_hash_cars()

    # Reset traffic phases at the begining of each period
    intersection = global_data_flow.m_unique_intersection
    phasetime = time - ECIGetStartingTimePhase(intersection)
    currentPhase = ECIGetCurrentPhase(intersection) - 1
    if (phasetime == 0 and currentPhase == 0):
        phase_num = 1
        for phaseLen in global_data_flow.intersection_list[0]["phase_duration"]:
            ECIChangeTimingPhase(
                        intersection, 
                        phase_num,
                        phaseLen, 
                        timeSta)
            phase_num += 1

    # calculate one layer of feature_grid_np
    segment_list = global_data_flow.get_segment_list()
    car_info_list = get_car_info(segment_list, [])
    # AKIPrintString("Manage1")
    global_data_flow.encode_car_list(time, car_info_list)
    # AKIPrintString("Manage2")
    # AKIPrintString(str(time))
    global_data_flow.update_status_grid(time)
    # AKIPrintString("Manage3")

    # for seg in global_data_flow.road_seg_arr:
    #     seg.get_bus_info()
    return 0


def decode_action():
    global threadCommMsg
    if threadCommMsg == "EXTEND":
        return True
    else:
        return False

def AAPIPostManage(time, timeSta, timTrans, SimStep):
    # print("[AIMSUN DEBUG] AAPIPostManage")
    global global_data_flow
    global g_logFile

    log_row_list = []
    log_row_list.append(str(time))

    intersection = global_data_flow.m_unique_intersection

    phasetime = time - ECIGetStartingTimePhase(intersection)
    currentPhase = ECIGetCurrentPhase(intersection) - 1

    log_row_list.append(str(currentPhase))
    log_row_list.append(str(phasetime))

    global_data_flow.post_update(time)

    for road_seg in global_data_flow.road_seg_arr:
        if not road_seg.is_has_bus_stop:
            continue
        if road_seg.last_stop_time != 0:
            new_dwell_time = 0.032 * (road_seg.last_stop_time + 4)
            # print(str(new_dwell_time))
            AKIPTVehModifyStopTime(road_seg.fastest_bus_id, 7, new_dwell_time)

    # print(global_data_flow.feature_grid_np.shape)

    grid_shape = global_data_flow.feature_grid_np.shape
    latest_status_grid = global_data_flow.status_grid_np[grid_shape[2] - 1]

    if grid_shape[2] >= 5:
        featureGrid = np.zeros((8, grid_shape[0], grid_shape[1]))
        # Five channels of cars' position grid
        for i in range(5):
            featureGrid[i] = global_data_flow.feature_grid_np[:,:,grid_shape[2] - 1 - i]
        
        # One channel of Signal Phase [HARD CODE HERE, please be advised]
        # Contain current phase and the time till the end of phase of interest
        featureGrid[5, max(global_data_flow.road_seg_arr[0].lane_num), int(math.floor(global_data_flow.road_seg_arr[0].total_len / GRID_SIZE) + 1)] = currentPhase
        featureGrid[5, max(global_data_flow.road_seg_arr[0].lane_num) + 1, int(math.floor(global_data_flow.road_seg_arr[0].total_len / GRID_SIZE) + 1)] = latest_status_grid[2]

        # One channel of dwell time
        # Mark at bus station
        for road_seg_idx in range(len(global_data_flow.road_seg_arr)):
            if not global_data_flow.road_seg_arr[road_seg_idx].is_has_bus_stop:
                continue
            if global_data_flow.road_seg_arr[road_seg_idx].dwell_time != 0:
                x, y = global_data_flow.hash_position(road_seg_idx, global_data_flow.road_seg_arr[road_seg_idx].bus_stop_distance_to_end, 0)
                featureGrid[6][y][x] = global_data_flow.road_seg_arr[road_seg_idx].dwell_time

        # One channel of time since last bus out
        # Mark at intersection
        # WB places at lane(number of EB lanes) of intersection
        # EB places at lane0 of intersection
        for road_seg_idx in range(len(global_data_flow.road_seg_arr)):
            x, y = global_data_flow.hash_position(0, 0, 0)
            if road_seg_idx == 0:
                featureGrid[7][0][x] = latest_status_grid[1] # EB since last bus out
            elif road_seg_idx == 1:
                featureGrid[7][max(global_data_flow.road_seg_arr[road_seg_idx].lane_num)][x] = latest_status_grid[5] # EB since last bus out

        with open(STATE_PATH, 'wb') as f:
            np.save(f, featureGrid)

        ## Check if notify neural network
        notify = False
        for road_seg_idx in range(len(global_data_flow.road_seg_arr)):
            road_seg = global_data_flow.road_seg_arr[road_seg_idx]
            if ((road_seg.bus_num != 0 or (abs(time - road_seg.last_bus_out)) < 0.1) and global_data_flow.head_way[road_seg_idx] != -1):
                notify = True
                # print("ROAD SEG DATA - " + str(road_seg.bus_num) + " " + str(road_seg.last_bus_out) + " " + str(global_data_flow.head_way[road_seg_idx]))
            log_row_list.append(str(road_seg.bus_num))
            log_row_list.append(str(road_seg.last_stop_time))
            log_row_list.append(str(road_seg.last_bus_out))
            log_row_list.append(str(global_data_flow.head_way[road_seg_idx]))
        
        # check current time is in interested phase zone
        if currentPhase not in global_data_flow.m_interested_phases:
            notify = False
        if currentPhase == 1 and phasetime <= global_data_flow.intersection_list[0]["phase_duration"][1] - 15:
            notify = False
        if currentPhase == 4 and phasetime <= global_data_flow.intersection_list[0]["phase_duration"][4] - 20:
            notify = False

        log_row_list.append(str(notify))
        # Corner case: Update last checkout when not being notified
        # This is for the case that a bus has state last time but get out before the next state
        if not notify:
            # If exactly one step after the notifying stage
            if time == global_data_flow.lastTimestepTime + 1:
                pass

            for road_seg_idx in range(len(global_data_flow.road_seg_arr)):
                # If there is no bus stopped by the traffic light. We don't give a damn care about the 
                # "through-through" bus (i.e. come in and go out without inside period of interest)
                if global_data_flow.lastTimestepBusNum[road_seg_idx] == 0:
                    continue

                last_bus_out = global_data_flow.road_seg_arr[road_seg_idx].last_bus_out
                # This is to cover the bus which comes in detected, but missed in check out.
                # Needs to fill the closing reward here.
                if (last_bus_out == time):
                    # print("Bus out not captured by reward")
                    if global_data_flow.busOutNotCapturedHeadway[road_seg_idx] == -1:
                        global_data_flow.busOutNotCapturedHeadway[road_seg_idx] = global_data_flow.head_way[road_seg_idx]

        ### All the usefull data are calculated, check if give rewards back and further notify neural network
        if global_data_flow.requestReward and notify:
            # Update the variables that need to be updated only within the step
            reward_list = []
            for road_seg_idx in range(len(global_data_flow.road_seg_arr)):
                busNum = global_data_flow.road_seg_arr[road_seg_idx].bus_num
                
                reward = 0
                log_row_list.append(str(global_data_flow.busOutNotCapturedHeadway[road_seg_idx]))
                # First count the situation where a bus checked out but not captured
                if (global_data_flow.busOutNotCapturedHeadway[road_seg_idx] != -1):
                    reward += abs(global_data_flow.lastTimestepHeadway[road_seg_idx] - TARGET_HEADWAY) -\
                            abs(global_data_flow.busOutNotCapturedHeadway[road_seg_idx] - TARGET_HEADWAY)

                if (busNum > 0):
                    if (global_data_flow.lastTimestepBusNum[road_seg_idx] == 0 or \
                        global_data_flow.lastTimestepBusNum[road_seg_idx] != 0 and global_data_flow.busOutNotCapturedHeadway[road_seg_idx] != -1):
                        if global_data_flow.head_way[road_seg_idx] > TARGET_HEADWAY:
                            reward += -1
                        else:
                            reward += 1
                    else:
                        reward += abs(global_data_flow.lastTimestepHeadway[road_seg_idx] - TARGET_HEADWAY) - abs(global_data_flow.head_way[road_seg_idx] - TARGET_HEADWAY)
                
                reward_list.append(reward)

            # print(reward_list)
            log_row_list.append(str(reward_list))
            log_row_list.append(str(global_data_flow.lastTimestepHeadway))
            total_reward = 0
            for road_seg_idx in range(len(global_data_flow.road_seg_arr)):
                if reward_list[road_seg_idx] == 0:
                    continue
                
                delta_total_reward = -1
                if (time - global_data_flow.lastTimestepTime > 1):

                    delta_total_reward = reward_list[road_seg_idx]
                elif (reward_list[road_seg_idx] >= 0.9):
                    delta_total_reward = 1
                elif (reward_list[road_seg_idx] <= -0.9):
                    delta_total_reward = -1

                log_row_list.append(str(delta_total_reward))
                if total_reward == 0:
                    total_reward = delta_total_reward
                elif reward_list[0] * reward_list[1] > 0:
                    total_reward += delta_total_reward
                elif reward_list[road_seg_idx] != 0:
                    dev0 = abs(global_data_flow.head_way[0] - TARGET_HEADWAY)
                    dev1 = abs(global_data_flow.head_way[1] - TARGET_HEADWAY)
                    if ((dev0 + dev1) != 0):
                        total_reward = dev0 / (dev0 + dev1) * total_reward + \
                                    dev1 / (dev0 + dev1) * delta_total_reward
                    else:
                        total_reward += delta_total_reward

            # print("[AIMSUN] Send reward info " + str(total_reward))
            sendmsg("REWARD - " + str(total_reward))
            log_row_list.append(str(total_reward))
        else:
            log_row_list.append(str("--"))

        # print("DATA - " + str(currentPhase) + " " + str(phasetime) + " " + str(time))
        if notify:
            global_data_flow.busOutNotCapturedHeadway = [-1, -1]
            # Update the variables that need to be updated only within the step
            busNumNow  = []
            for road_seg_idx in range(len(global_data_flow.road_seg_arr)):
                busNum = global_data_flow.road_seg_arr[road_seg_idx].bus_num
                busNumNow.append(busNum)
            
            global_data_flow.lastTimestepTime = time
            global_data_flow.lastTimestepBusNum = busNumNow
            global_data_flow.lastTimestepHeadway = [i for i in global_data_flow.head_way] # Need to perform deep copy
            global_data_flow.requestReward = True

            # print("Bus in interested range and in interested phase " + str(time))
            globalLock.acquire()
            # print("[Aimsun] Send DATA_READY")
            sendmsg("DATA_READY - " + str(time))
            # wait for action ready

            # globalLock.acquire()
            

            # Check for possible traffic light extension
            pdur = doublep()
            pcmax = doublep()
            pcmin = doublep()
            ECIGetDurationsPhase(intersection, currentPhase + 1, timeSta, pdur, pcmax, pcmin)
            extandAction = decode_action()
            if extandAction:
                log_row_list.append(str("EXTEND"))
                # Update traffic light phase time if needed
                if phasetime >= int(pdur.value()):
                    originalPhaseLength = global_data_flow.intersection_list[0]["phase_duration"][currentPhase]
                    # print(int(pdur.value()), originalPhaseLength)

                    # Phase one is allowed to extend 10 seconds [HARD CODE]
                    if (int(pdur.value()) - originalPhaseLength < 10 and currentPhase == 1):
                        ECIChangeTimingPhase(
                        intersection, 
                        currentPhase + 1,
                        int(pdur.value()) + 1, 
                        timeSta)
                    elif (int(pdur.value()) - originalPhaseLength < 20 and currentPhase == 4):
                        # Phase four is allowed to extend 20 seconds [HARD CODE]
                        ECIChangeTimingPhase(
                        intersection, 
                        currentPhase + 1,
                        int(pdur.value()) + 1, 
                        timeSta)
            else:
                log_row_list.append(str("NOTHING"))
                # stop the phase and go to the next state immediately
                ECIChangeTimingPhase(
                    intersection, 
                    currentPhase + 1,
                    phasetime, 
                    timeSta)

            if currentPhase == 0 and phasetime == 1:
                # Reset all the phase traffic light length
                for phase_idx in range(len(global_data_flow.intersection_list[0]["phase_duration"])):
                    ECIChangeTimingPhase(
                        intersection, 
                        phase_idx + 1,
                        global_data_flow.intersection_list[0]["phase_duration"][phase_idx], 
                        timeSta)
                
        # else:
        #     global_data_flow.requestReward = False
            # print("no notification")
    # else:
    #     print("Data not ready. Continue...")

    # print("====== Cycle End ======")
    g_logFile.write(",".join(log_row_list) + "\n")
    return 0


def AAPIFinish():
    global global_data_flow

    global g_logFile
    g_logFile.close()
    
    for i in range(2):
        since_last_bus_stop_zero = -1
        since_last_bus_out_zero = -1
        for t in range(global_data_flow.status_grid_np.shape[0]):
            if global_data_flow.status_grid_np[t][i * 4] == 0:
                if global_data_flow.status_grid_np[since_last_bus_stop_zero + 1][i * 4] != -1:
                    max_wait_time = global_data_flow.status_grid_np[t - 1][i * 4]
                    for j in range(since_last_bus_stop_zero + 1, t):
                        global_data_flow.status_grid_np[j][i * 4] = max_wait_time - global_data_flow.status_grid_np[j][i * 4] + 1
                since_last_bus_stop_zero = t
            if global_data_flow.status_grid_np[t][i * 4 + 1] == 0:
                if global_data_flow.status_grid_np[since_last_bus_out_zero + 1][i * 4 + 1] != -1:
                    max_wait_time = global_data_flow.status_grid_np[t - 1][i * 4 + 1]
                    for j in range(since_last_bus_out_zero + 1, t):
                        global_data_flow.status_grid_np[j][i * 4 + 1] = max_wait_time - global_data_flow.status_grid_np[j][i * 4 + 1] + 1
                since_last_bus_out_zero = t

        for t in range(since_last_bus_stop_zero, global_data_flow.status_grid_np.shape[0]):
            global_data_flow.status_grid_np[t][i * 4] = -1
        
        for t in range(since_last_bus_out_zero, global_data_flow.status_grid_np.shape[0]):
            global_data_flow.status_grid_np[t][i * 4 + 1] = -1


    # for t in range(global_data_flow.status_grid_np.shape[0]):
    #     print(str(global_data_flow.status_grid_np[t]))

    # with open('C:\\Users\\Public\\Documents\\ShalabyGroup\\TSP-Louis\\history\\' + str(ANGConnGetReplicationId()) + '_feature.npy', 'wb') as f:
    #     np.save(f, global_data_flow.feature_grid_np)
    # with open('C:\\Users\\Public\\Documents\\ShalabyGroup\\TSP-Louis\\history\\' + str(ANGConnGetReplicationId()) + '_status.npy', 'wb') as f:
    #     np.save(f, global_data_flow.status_grid_np)
    # with open('C:\\Users\\Public\\Documents\\ShalabyGroup\\TSP-Louis\\history\\' + str(ANGConnGetReplicationId()) + '_headway.npy', 'wb') as f:
    #     np.save(f, np.array(global_data_flow.head_way))

    # with open('C:\\Users\\siwei\\Documents\\Developer\\aimsun\\' + str(ANGConnGetReplicationId()) + '_feature.npy', 'wb') as f:
    #     np.save(f, global_data_flow.feature_grid_np)
    # with open('C:\\Users\\siwei\\Documents\\Developer\\aimsun\\' + str(ANGConnGetReplicationId()) + '_status.npy', 'wb') as f:
    #     np.save(f, global_data_flow.status_grid_np)
    # with open('C:\\Users\\siwei\\Documents\\Developer\\aimsun\\' + str(ANGConnGetReplicationId()) + '_headway.npy', 'wb') as f:
    #     np.save(f, np.array(global_data_flow.head_way))
    return 0


def AAPIUnLoad():
    AKIPrintString("Finished")
    sendmsg("FIN")

    return 0


def AAPIEnterVehicle(idveh, idsection):

    return 0


def AAPIExitVehicle(idveh, idsection):

    return 0
