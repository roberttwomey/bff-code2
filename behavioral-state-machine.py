import time
import sys
import os
import json
import asyncio
import enum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, UwbState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_, unitree_go_msg_dds__UwbState_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.rpc.client import Client
from unitree_sdk2py.utils.crc import CRC


# import unitree_legged_const as go2
class Go2Constants:
    LegID = {
        "FR_0": 0,  # Front right hip
        "FR_1": 1,  # Front right thigh
        "FR_2": 2,  # Front right calf
        "FL_0": 3,
        "FL_1": 4,
        "FL_2": 5,
        "RR_0": 6,
        "RR_1": 7,
        "RR_2": 8,
        "RL_0": 9,
        "RL_1": 10,
        "RL_2": 11,
    }
    
    HIGHLEVEL = 0xEE
    LOWLEVEL = 0xFF
    TRIGERLEVEL = 0xF0
    PosStopF = 2.146e9
    VelStopF = 16000.0

go2 = Go2Constants()

ethernet_interface = "enP8p1s0"


def is_interface_active(interface_name):
    """Check if the network interface exists and is active (UP)."""
    operstate_path = f"/sys/class/net/{interface_name}/operstate"
    if not os.path.exists(operstate_path):
        return False
    
    try:
        with open(operstate_path, 'r') as f:
            operstate = f.read().strip()
        return operstate == "up"
    except (IOError, OSError):
        return False


class RobotState(enum.Enum):
    WALKING_MODE = "walking"
    SPEAKING_MODE = "speaking"
    THINKING_MODE = "thinking"
    DREAMING_MODE = "dreaming"
    POWER_OFF = "power_off"

class BehavioralStateMachine:
    def __init__(self):
        # Initialize components
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self.InitLowCmd()
        
        # Sport client for robot commands
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # Motion switcher client
        self.motion_switcher = MotionSwitcherClient()
        self.motion_switcher.SetTimeout(10.0)
        self.motion_switcher.Init()

        # Robot state client
        self.robot_state_client = RobotStateClient()
        self.robot_state_client.SetTimeout(10.0)
        self.robot_state_client.Init()
        
        # VUI client for LED colors
        self.vui_client = Client('vui')
        self.vui_client.SetTimeout(3.0)
        self.vui_client._RegistApi(1007, 0)
        
        # Lidar publisher
        self.lidar_publisher = ChannelPublisher("rt/utlidar/switch", String_)
        self.lidar_publisher.Init()
        self.lidar_cmd = std_msgs_msg_dds__String_()
        
        # State management
        self.current_state = RobotState.POWER_OFF  # None indicates no state has been set yet
        self.last_controller_activity_time = time.time()
        self.last_uwb_activity_time = time.time()
        self.idle_timeout = 45  # seconds
        self.monitoring = True
        
        # Wireless controller state
        self.controller_active = False
        self.uwb_active = False
        self.last_controller_data = None
        
        # Subscribe to low state for controller input
        self.lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        
        # Subscribe to UWB state for UWB controller input
        self.uwb_subscriber = ChannelSubscriber("rt/uwbstate", UwbState_)
        self.uwb_subscriber.Init(self.UwbStateMessageHandler, 10)
        
    def InitLowCmd(self):
        """Initialize LowCmd with required header and motor values"""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        
        # Initialize all motor commands to safe values
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
    
    def LowStateMessageHandler(self, msg: LowState_):
        """Handle incoming low state messages, including wireless controller data"""
        self.last_controller_data = msg
        self.check_controller_activity(msg)
    
    def check_controller_activity(self, msg: LowState_):
        """Check if wireless controller is being used"""
        try:
            wireless_data = msg.wireless_remote
            if wireless_data:
                # Check if any joystick values are non-zero or buttons are pressed
                # Extract float values for joysticks
                import struct
                lx = struct.unpack('<f', wireless_data[4:8])[0] if len(wireless_data) > 7 else 0
                rx = struct.unpack('<f', wireless_data[8:12])[0] if len(wireless_data) > 11 else 0
                ry = struct.unpack('<f', wireless_data[12:16])[0] if len(wireless_data) > 15 else 0
                ly = struct.unpack('<f', wireless_data[20:24])[0] if len(wireless_data) > 23 else 0
                
                # Check button presses
                data1 = wireless_data[2] if len(wireless_data) > 2 else 0
                data2 = wireless_data[3] if len(wireless_data) > 3 else 0
                
                # Determine if controller is active
                self.controller_active = (
                    abs(lx) > 0.01 or abs(rx) > 0.01 or abs(ry) > 0.01 or abs(ly) > 0.01 or
                    data1 != 0 or data2 != 0
                )
                
                if self.controller_active:
                    if time.time() - self.last_controller_activity_time > 1:
                        print(f"Controller activity detected - Joysticks: Lx={lx:.2f}, Rx={rx:.2f}, Ry={ry:.2f}, Ly={ly:.2f}")
                    self.last_controller_activity_time = time.time()
                    
        except Exception as e:
            print(f"Error checking controller activity: {e}")
    
    def UwbStateMessageHandler(self, msg: UwbState_):
        """Handle incoming UWB state messages, including joystick data"""
        try:
            # Get joystick values from UWB state
            joystick = msg.joystick if hasattr(msg, 'joystick') else [0.0, 0.0]
            buttons = msg.buttons if hasattr(msg, 'buttons') else 0
            
            # Check if joystick is being moved or buttons are pressed
            # Joystick is a list/array with [x, y] values
            jx = joystick[0] if len(joystick) > 0 else 0.0
            jy = joystick[1] if len(joystick) > 1 else 0.0
            
            # Determine if UWB controller is active
            self.uwb_active = (
                abs(jx) > 0.01 or abs(jy) > 0.01 or buttons != 0
            )
            
            if self.uwb_active:
                if time.time() - self.last_uwb_activity_time > 1:
                    print(f"UWB controller activity detected - Joystick: X={jx:.2f}, Y={jy:.2f}, Buttons={buttons}")
                self.last_uwb_activity_time = time.time()
                
        except Exception as e:
            print(f"Error checking UWB controller activity: {e}")
    
    def set_vui_color(self, color, duration=0):
        """Set the VUI LED color"""
        p = {}
        p["color"] = color
        p["time"] = duration
        parameter = json.dumps(p)
        
        code, result = self.vui_client._Call(1007, parameter)
        
        if code != 0:
            print(f"Set color error. code: {code}, {result}")
            return False
        else:
            print(f"Set color {color} success")
            return True
    
    def set_lidar_state(self, status):
        """Set lidar on or off"""
        if status == "OFF":
            self.lidar_cmd.data = "OFF"
        elif status == "ON":
            self.lidar_cmd.data = "ON"
        else:
            print(f"Invalid lidar status: {status}")
            return False
        
        try:
            self.lidar_publisher.Write(self.lidar_cmd)
            print(f"Lidar set to {status}")
            return True
        except Exception as e:
            print(f"Error setting lidar to {status}: {e}")
            return False
    
    def transition_to_state(self, new_state: RobotState):
        """Transition to a new state and update VUI color"""
        if self.current_state == new_state:
            return
        
        print(f"Transitioning from {self.current_state.value} to {new_state.value}")
        self.current_state = new_state
        
        # Update VUI color based on state
        color_map = {
            RobotState.WALKING_MODE: "green",
            RobotState.SPEAKING_MODE: "green",
            RobotState.THINKING_MODE: "purple",
            RobotState.DREAMING_MODE: "cyan",
            RobotState.POWER_OFF: "red"  # Will turn off
        }
        
        color = color_map.get(new_state, "green")
        self.set_vui_color(color, duration=9999)  # duration=0 for persistent
    
    def stand_down(self):
        """Make robot stand down"""
        print("Standing down...")
        try:
            self.sport_client.StandDown()
            print("StandDown command sent")
        except Exception as e:
            print(f"Error in StandDown: {e}")
    
    def stand_up(self):
        """Make robot stand up"""
        print("Standing up...")
        try:
            self.sport_client.StandUp()
            print("StandUp command sent")
        except Exception as e:
            print(f"Error in StandUp: {e}")

    def balance_stand(self):
        """Make robot balance stand"""
        print("Entering balance stand...")
        try:
            self.sport_client.BalanceStand()
            print("BalanceStand command sent")
        except Exception as e:
            print(f"Error in BalanceStand: {e}")
    
    def release_mcf_mode(self):
        """Release MCF mode service"""
        print("Checking and releasing MCF mode...")
        try:
            status, result = self.motion_switcher.CheckMode()
            
            if result['name']:
                print(f"Current mode: {result['name']}, releasing...")
                self.motion_switcher.ReleaseMode()
                time.sleep(1)
                
                # Verify mode was released
                status, result = self.motion_switcher.CheckMode()
                if result['name']:
                    print(f"Warning: Mode still active: {result['name']}")
                else:
                    print("MCF mode released successfully")
            else:
                print("No active mode found")
        except Exception as e:
            print(f"Error releasing MCF mode: {e}")
    
    def start_mcf_service(self):
        """Start MCF mode service"""
        print("Starting MCF mode service...")
        self.robot_state_client.ServiceSwitch("mcf", True)
        time.sleep(2)
        print("MCF mode service started successfully")

    def ensure_mcf_mode(self):
        """Ensure robot is in MCF mode"""

        self.start_mcf_service()
        time.sleep(2)

        print("Ensuring MCF mode...")
        try:
            status, result = self.motion_switcher.CheckMode()
            if result['name'] != 'mcf':
                print(f"Current mode: {result['name']}, switching to MCF...")
                self.motion_switcher.SelectMode("mcf")
                time.sleep(2)
            else:
                print("Already in MCF mode")
        except Exception as e:
            print(f"Error ensuring MCF mode: {e}")
    
    def send_bms_off_command(self):
        """Send BMS command to turn off (off=0xA5)"""
        print("Sending BMS off command...")
        try:
            self.low_cmd.bms_cmd.off = 0xA5
            self.low_cmd.bms_cmd.reserve = [0, 0, 0]
            
            # Calculate and set CRC
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            
            # Publish the command
            ret = self.lowcmd_publisher.Write(self.low_cmd)
            print(f"BMS off command published. Return value: {ret}")
        except Exception as e:
            print(f"Error sending BMS off command: {e}")
    
    def check_idle_state(self):
        """Check if robot has been idle and transition to DREAMING if needed"""
        if not self.monitoring:
            return
        
        if self.current_state == RobotState.POWER_OFF:
            return
        
        current_time = time.time()
        idle_duration = min(current_time - self.last_controller_activity_time, current_time - self.last_uwb_activity_time)
        
        # If in WALKING mode and idle for too long, transition to DREAMING
        if self.current_state == RobotState.WALKING_MODE and idle_duration >= self.idle_timeout:
            print(f"Robot idle for {idle_duration:.1f}s. Transitioning to DREAMING...")
            # Turn off lidar before going prone
            self.transition_to_state(RobotState.DREAMING_MODE)
            self.stand_down()
            time.sleep(2)
            self.set_lidar_state("OFF")

    
    def check_wake_from_dreaming(self):
        """Check if controller activity should wake from DREAMING to WALKING"""
        if self.current_state == RobotState.DREAMING_MODE and (self.controller_active or self.uwb_active):
            print("Controller activity detected while dreaming. Waking up to WALKING mode...")
            
            # Ensure MCF mode is active before standing
            # self.ensure_mcf_mode()
            
            # Turn on lidar before standing up
            self.set_lidar_state("ON")
            time.sleep(1)

            self.transition_to_state(RobotState.WALKING_MODE)

            self.stand_up()
            time.sleep(2)
            
            self.balance_stand()
            time.sleep(2)

            self.last_controller_activity_time = time.time()
            self.last_uwb_activity_time = time.time()
    
    def start(self):
        """Initialize the robot to WALKING mode"""
        print("Starting Behavioral State Machine...")
        print(f"Idle timeout: {self.idle_timeout}s")
        
        # Turn on lidar initially
        self.set_lidar_state("ON")
        time.sleep(2)

        # Initial state: WALKING_MODE
        self.transition_to_state(RobotState.WALKING_MODE)
        self.last_activity_time = time.time()

        # Ensure MCF mode is active
        self.ensure_mcf_mode()
        time.sleep(2)
                
        # Make robot stand up
        self.stand_up()
        time.sleep(2)

        # Start balance stand
        self.balance_stand()
        time.sleep(2)
    
    def run_state_machine(self):
        """Main state machine loop"""
        print("State machine running. Press Ctrl+C to initiate POWER_OFF...")
        
        try:
            while self.monitoring:
                # Check idle state (WALKING -> DREAMING)
                self.check_idle_state()
                
                # Check wake from dreaming (DREAMING -> WALKING)
                self.check_wake_from_dreaming()
                
                # Display current state
                idle_duration = min(time.time() - self.last_controller_activity_time, time.time() - self.last_uwb_activity_time)
                # idle_duration = time.time() - self.last_ctivity_time
                print(f"\rState: {self.current_state.value.upper():15} | Idle: {idle_duration:.1f}s", end="")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\nCtrl+C detected. Initiating POWER_OFF sequence...")
            self.transition_to_state(RobotState.POWER_OFF)
            
            # Stand down
            self.stand_down()
            time.sleep(2)

            # Release MCF mode
            self.release_mcf_mode()
            time.sleep(2)
            
            # Power off
            # self.send_bms_off_command()
            # time.sleep(2)
            
            # print("Power off sequence complete.")
            # self.monitoring = False
    
    def set_thinking_mode(self):
        """Manually set to thinking mode"""
        self.transition_to_state(RobotState.THINKING_MODE)
        self.last_activity_time = time.time()
    
    def set_speaking_mode(self):
        """Manually set to speaking mode"""
        self.transition_to_state(RobotState.SPEAKING_MODE)
        self.last_activity_time = time.time()


if __name__ == '__main__':
    print("WARNING: This script controls robot behavior and will power off the robot when Ctrl+C is pressed!")
    # print("Ensure there are no obstacles around the robot.")
    # input("Press Enter to continue...")
    
    if len(sys.argv) > 1:
        interface_name = sys.argv[1]
    else:
        interface_name = ethernet_interface

    # Check if ethernet interface is active first
    if not is_interface_active(interface_name):
        print(f"not connected: interface {interface_name} is not active")
        sys.exit(0)
    
    # Initialize channel factory
    ChannelFactoryInitialize(0, interface_name)
    
    # Set up subscriber to check connection status
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    message_received = [False]
    
    def LowStateHandler(msg: LowState_):
        message_received[0] = True
        
    sub.Init(LowStateHandler, 10)
    
    # Wait for a message with a timeout (2 seconds)
    timeout = 2.0
    start_time = time.time()
    
    while not message_received[0] and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    if not message_received[0]:
        print(f"not connected: message not received on interface {interface_name}")
        sys.exit(0)
        
    print(f"connected: message received on interface {interface_name}")
    
    state_machine = BehavioralStateMachine()
    state_machine.start()
    
    # Run the state machine
    state_machine.run_state_machine()
    
    print("Program ended.")

