import time
import sys
import subprocess
import argparse

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, BmsCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__BmsCmd_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

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

ethernet_interface = "enP8p1s0"

class Custom:
    def __init__(self):
        # create publisher for low-level commands
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        
        # Initialize CRC calculator
        self.crc = CRC()
        
        # Initialize the low command with proper headers
        self.InitLowCmd()
        
        # create sport client for stand down
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # create motion switcher client to release MFC mode
        self.motion_switcher = MotionSwitcherClient()
        self.motion_switcher.SetTimeout(10.0)
        self.motion_switcher.Init()

        self.robot_state_client = RobotStateClient()
        self.robot_state_client.SetTimeout(10.0)
        self.robot_state_client.Init()

        # Lidar publisher
        self.lidar_publisher = ChannelPublisher("rt/utlidar/switch", String_)
        self.lidar_publisher.Init()
        self.lidar_cmd = std_msgs_msg_dds__String_()
    
    def InitLowCmd(self):
        """Initialize LowCmd with required header and motor values"""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        
        # Initialize all motor commands to safe values
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
        
    def set_lidar_state(self, status: str) -> bool:
        """
        Set lidar on or off.
        
        Args:
            status: "ON" or "OFF"
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if status not in ["ON", "OFF"]:
                print(f"Invalid lidar status: {status}")
                return False
            
            self.lidar_cmd.data = status
            self.lidar_publisher.Write(self.lidar_cmd)
            return True
        except Exception as e:
            print(f"Error setting lidar state: {e}")
            return False
        
    def stand_down(self):
        """Asking robot to stand down"""
        print("Asking robot to stand down...")
        self.set_lidar_state("OFF")
        time.sleep(1.0)
        self.sport_client.StandDown()
    
    def stand_up(self):
        """Asking robot to stand up"""
        print("Asking robot to stand up...")
        self.sport_client.StandUp()

    def damp_mode(self):
        """Asking robot to damp mode"""
        print("Asking robot to damp mode...")
        self.sport_client.Damp()
    
    def balance_stand(self):
        """Asking robot to balance stand"""
        print("Asking robot to balance stand...")
        self.sport_client.BalanceStand()

    def release_mcf_mode(self):
        """Release MCF mode service"""
        print("Checking and releasing MCF mode...")
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
                print(f"MCF mode released successfully {result}")
        else:
            print("No active mode found, continuing...")
        
    def list_services(self):
        """List all services"""
        print("Listing all services...")

        code, result = self.robot_state_client.ServiceList()

        for service in result:
            print(f"Service: {service.name}, Status: {service.status}, Protect: {service.protect}")
        # print(f"Services: {result}")
        
    def start_mcf_service(self):
        """Start MCF mode service"""
        print("Starting MCF mode service...")

        result = self.robot_state_client.ServiceSwitch("mcf", True)
        if result != 0:
            print(f"Error starting MCF mode service: {result}")
        else:
            print(f"MCF mode service started successfully: {result}")

        print("Checking MCF mode...")
        status, result = self.motion_switcher.CheckMode()

        if result['name']:
            print(f"Current mode: {result['name']}...")

        else:
            print("No active mode found, starting MCF mode...")

        print("Starting MCF mode...")
        self.motion_switcher.SelectMode("mcf")
        
        # Verify mode was started
        status, result = self.motion_switcher.CheckMode()
        if result['name']:
                print(f"MCF mode started successfully: {result['name']}")
        else:
            print("Warning: MCF mode not started")
        

    def send_bms_off_command(self):
        """Send BMS command to turn off (off=0xA5)"""
        print("Sending BMS off command...")
        # Set the bms_cmd field to turn off

        self.low_cmd.bms_cmd.off = 0xA5
        # Set reserve field (3 bytes of zeros)
        self.low_cmd.bms_cmd.reserve = [0, 0, 0]
        
        # Calculate and set CRC (REQUIRED before publishing!)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        
        # Publish the command
        ret = self.lowcmd_publisher.Write(self.low_cmd)
        print("BMS off command published. Return value: ", ret)
        

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("This will release MFC mode, make the robot stand down, and send a power off command!")
    input("Press Enter to continue...")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot shutdown script')
    parser.add_argument('--yes', action='store_true', 
                       help='Required flag to actually execute BMS power off and system shutdown')
    parser.add_argument('interface', nargs='?', default=ethernet_interface,
                       help='Ethernet interface (default: enP8p1s0)')
    args = parser.parse_args()

    # Initialize channel factory with interface
    ChannelFactoryInitialize(0, args.interface)

    custom = Custom()
    
    # Step 1: Make the robot stand down
    custom.stand_down()
    time.sleep(2)

    # Step 2: Release MFC mode service
    custom.release_mcf_mode()
    time.sleep(2)
        
    # custom.list_services()
    # time.sleep(2)

    # Step 3: Start MFC mode service
    # custom.start_mcf_service()
    # time.sleep(2)

    # custom.list_services()
    # time.sleep(2)

    # # Step 3: Stand up
    # custom.stand_up()
    # time.sleep(2)
   
    # # Step 4: Balance stand
    # custom.balance_stand()
    # time.sleep(2)

    # Step 3: Send the BMS off command (only if --yes flag is provided)
    if args.yes:
        custom.send_bms_off_command()
        
        # Small delay to ensure BMS command is transmitted over network before host shutdown
        time.sleep(0.5)
        
        # Step 4: Execute system shutdown immediately
        print("Executing system shutdown now...")
        # Try systemctl first (better for systemd systems), fallback to shutdown
        result = subprocess.run(["systemctl", "poweroff"], check=False, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback to shutdown command
            subprocess.run(["shutdown", "-h", "now"], check=False)
        
        time.sleep(2)
        print("Commands sent successfully.")
    else:
        print("\n" + "="*60)
        print("SAFETY CHECK: BMS power off and system shutdown were NOT executed.")
        print("To actually power off the robot, run with --yes flag:")
        print(f"  python {sys.argv[0]} --yes")
        print("="*60)
        print("Stand down and MFC mode release completed successfully.")


