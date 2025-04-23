import json
import os
import platform
import re
import shutil
import time
import requests
from logzero import logger
from tqdm import tqdm
import socket
from urllib.request import urlopen
import ssl
import xlwt
import psutil
import signal
import cv2
from functools import wraps
from jinja2 import Environment, FileSystemLoader
from tidevice._device import Device
from tidevice import Usbmux
from solox.public.adb import adb


class Platform:
    Android = 'Android'
    iOS = 'iOS'
    Mac = 'MacOS'
    Windows = 'Windows'

class Devices:

    def __init__(self, platform=Platform.Android):
        self.platform = platform
        self.adb = adb.adb_path

    def execCmd(self, cmd):
        """Execute the command to get the terminal print result"""
        r = os.popen(cmd)
        try:
            text = r.buffer.read().decode(encoding='gbk').replace('\x1b[0m','').strip()
        except UnicodeDecodeError:
            text = r.buffer.read().decode(encoding='utf-8').replace('\x1b[0m','').strip()
        finally:
            r.close()
        return text

    def filterType(self):
        """Select the pipe filtering method according to the system"""
        filtertype = ('grep', 'findstr')[platform.system() == Platform.Windows]
        return filtertype

    def getDeviceIds(self):
        """Get all connected device ids"""
        Ids = list(os.popen(f"{self.adb} devices").readlines())
        deviceIds = []
        for i in range(1, len(Ids) - 1):
            id, state = Ids[i].strip().split()
            if state == 'device':
                deviceIds.append(id)
        return deviceIds

    def getDevicesName(self, deviceId):
        """Get the device name of the Android corresponding device ID"""
        try:
            devices_name = os.popen(f'{self.adb} -s {deviceId} shell getprop ro.product.model').readlines()[0].strip()
        except Exception:
            devices_name = os.popen(f'{self.adb} -s {deviceId} shell getprop ro.product.model').buffer.readlines()[0].decode("utf-8").strip()
        return devices_name

    def getDevices(self):
        """Get all Android devices"""
        DeviceIds = self.getDeviceIds()
        Devices = [f'{id}({self.getDevicesName(id)})' for id in DeviceIds]
        logger.info('Connected devices: {}'.format(Devices))
        return Devices

    def getIdbyDevice(self, deviceinfo, platform):
        """Obtain the corresponding device id according to the Android device information"""
        if platform == Platform.Android:
            deviceId = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", deviceinfo)
            if deviceId not in self.getDeviceIds():
                raise Exception('no device found')
        else:
            deviceId = deviceinfo
        return deviceId
    
    def getSdkVersion(self, deviceId):
        version = adb.shell(cmd='getprop ro.build.version.sdk', deviceId=deviceId)
        return version
    
    def getCpuCores(self, deviceId):
        """get Android cpu cores"""
        cmd = 'cat /sys/devices/system/cpu/online'
        result = adb.shell(cmd=cmd, deviceId=deviceId)
        try:
            nums = int(result.split('-')[1]) + 1
        except:
            nums = 1
        return nums

    def getPid(self, deviceId, pkgName):
        """Get the pid corresponding to the Android package name"""
        try:
            sdkversion = self.getSdkVersion(deviceId)
            if sdkversion and int(sdkversion) < 26:
                result = os.popen(f"{self.adb} -s {deviceId} shell ps | {self.filterType()} {pkgName}").readlines()
                processList = ['{}:{}'.format(process.split()[1],process.split()[8]) for process in result]
            else:
                result = os.popen(f"{self.adb} -s {deviceId} shell ps -ef | {self.filterType()} {pkgName}").readlines()
                processList = ['{}:{}'.format(process.split()[1],process.split()[7]) for process in result]
            for i in range(len(processList)):
                if processList[i].count(':') == 1:
                    index = processList.index(processList[i])
                    processList.insert(0, processList.pop(index))
                    break
            if len(processList) == 0:
               logger.warning('{}: no pid found'.format(pkgName))     
        except Exception as e:
            processList = []
            logger.exception(e)
        return processList

    def checkPkgname(self, pkgname):
        flag = True
        replace_list = ['com.google']
        for i in replace_list:
            if i in pkgname:
                flag = False
        return flag

    def getPkgname(self, deviceId):
        """Get all package names of Android devices"""
        pkginfo = os.popen(f"{self.adb} -s {deviceId} shell pm list packages --user 0")
        pkglist = [p.lstrip('package').lstrip(":").strip() for p in pkginfo]
        if pkglist.__len__() > 0:
            return pkglist
        else:
            pkginfo = os.popen(f"{self.adb} -s {deviceId} shell pm list packages")
            pkglist = [p.lstrip('package').lstrip(":").strip() for p in pkginfo]
            return pkglist

    def getDeviceInfoByiOS(self):
        """Get a list of all successfully connected iOS devices"""
        deviceInfo = [udid for udid in Usbmux().device_udid_list()]
        logger.info('Connected devices: {}'.format(deviceInfo))    
        return deviceInfo

    def getPkgnameByiOS(self, udid):
        """Get all package names of the corresponding iOS device"""
        d = Device(udid)
        pkgNames = [i.get("CFBundleIdentifier") for i in d.installation.iter_installed(app_type="User")]
        return pkgNames
    
    def get_pc_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            logger.error('get local ip failed')
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip
    
    def get_device_ip(self, deviceId):
        content = os.popen(f"{self.adb} -s {deviceId} shell ip addr show wlan0").read()
        logger.info(content)
        math_obj = re.search(r'inet\s(\d+\.\d+\.\d+\.\d+).*?wlan0', content)
        if math_obj and math_obj.group(1):
            return math_obj.group(1)
        return None
    
    def devicesCheck(self, platform, deviceid=None, pkgname=None):
        """Check the device environment"""
        match(platform):
            case Platform.Android:
                if len(self.getDeviceIds()) == 0:
                    raise Exception('no devices found')
                if len(self.getPid(deviceId=deviceid, pkgName=pkgname)) == 0:
                    raise Exception('no process found')
            case Platform.iOS:
                if len(self.getDeviceInfoByiOS()) == 0:
                    raise Exception('no devices found')
            case _:
                raise Exception('platform must be Android or iOS')        
            
    def getDdeviceDetail(self, deviceId, platform):
        result = dict()
        match(platform):
            case Platform.Android:
                result['brand'] = adb.shell(cmd='getprop ro.product.brand', deviceId=deviceId)
                result['name'] = adb.shell(cmd='getprop ro.product.model', deviceId=deviceId)
                result['version'] = adb.shell(cmd='getprop ro.build.version.release', deviceId=deviceId)
                result['serialno'] = adb.shell(cmd='getprop ro.serialno', deviceId=deviceId)
                cmd = f'ip addr show wlan0 | {self.filterType()} link/ether'
                wifiadr_content = adb.shell(cmd=cmd, deviceId=deviceId)                
                result['wifiadr'] = Method._index(wifiadr_content.split(), 1, '')
                result['cpu_cores'] = self.getCpuCores(deviceId)
                result['physical_size'] = adb.shell(cmd='wm size', deviceId=deviceId).replace('Physical size:','').strip()
            case Platform.iOS:
                ios_device = Device(udid=deviceId)
                result['brand'] = ios_device.get_value("DeviceClass", no_session=True)
                result['name'] = ios_device.get_value("DeviceName", no_session=True)
                result['version'] = ios_device.get_value("ProductVersion", no_session=True)
                result['serialno'] = deviceId
                result['wifiadr'] = ios_device.get_value("WiFiAddress", no_session=True)
                result['cpu_cores'] = 0
                result['physical_size'] = self.getPhysicalSzieOfiOS(deviceId)
            case _:
                raise Exception('{} is undefined'.format(platform)) 
        return result
    
    def getPhysicalSzieOfiOS(self, deviceId):
        ios_device = Device(udid=deviceId)
        try:
            screen_info = ios_device.screen_info()
            PhysicalSzie = '{}x{}'.format(screen_info.get('width'), screen_info.get('height'))
        except Exception as e:
            PhysicalSzie = ''  
            logger.exception(e)  
        return PhysicalSzie
    
    def getCurrentActivity(self, deviceId):
        result = adb.shell(cmd='dumpsys window | {} mCurrentFocus'.format(self.filterType()), deviceId=deviceId)
        if result.__contains__('mCurrentFocus'):
            activity = str(result).split(' ')[-1].replace('}','') 
            return activity
        else:
            raise Exception('No activity found')

    def getStartupTimeByAndroid(self, activity, deviceId):
        result = adb.shell(cmd='am start -W {}'.format(activity), deviceId=deviceId)
        return result

    def getStartupTimeByiOS(self, pkgname):
        try:
            import ios_device
        except ImportError:
            logger.error('py-ios-devices not found, please run [pip install py-ios-devices]') 
        result = self.execCmd('pyidevice instruments app_lifecycle -b {}'.format(pkgname))       
        return result          

class File:

    def __init__(self, fileroot='.'):
        self.fileroot = fileroot
        self.report_dir = self.get_repordir()

    def clear_file(self):
        logger.info('Clean up useless files ...')
        if os.path.exists(self.report_dir):
            for f in os.listdir(self.report_dir):
                filename = os.path.join(self.report_dir, f)
                if f.split(".")[-1] in ['log', 'json', 'mkv']:
                    os.remove(filename)
        Scrcpy.stop_record()            
        logger.info('Clean up useless files success')            

    def export_excel(self, platform, scene):
        logger.info('Exporting excel ...')
        android_log_file_list = ['cpu_app','cpu_sys','mem_total','mem_swap',
                                 'battery_level', 'battery_tem','upflow','downflow','fps','gpu']
        ios_log_file_list = ['cpu_app','cpu_sys', 'mem_total', 'battery_tem', 'battery_current', 
                             'battery_voltage', 'battery_power','upflow','downflow','fps','gpu']
        log_file_list = android_log_file_list if platform == 'Android' else ios_log_file_list
        wb = xlwt.Workbook(encoding = 'utf-8')
        k = 1
        for name in log_file_list:
            ws1 = wb.add_sheet(name)
            ws1.write(0,0,'Time') 
            ws1.write(0,1,'Value')
            row = 1 #start row
            col = 0 #start col
            if os.path.exists(f'{self.report_dir}/{scene}/{name}.log'):
                f = open(f'{self.report_dir}/{scene}/{name}.log','r',encoding='utf-8')
                for lines in f: 
                    target = lines.split('=')
                    k += 1
                    for i in range(len(target)):
                        ws1.write(row, col ,target[i])
                        col += 1
                    row += 1
                    col = 0
        xls_path = os.path.join(self.report_dir, scene, f'{scene}.xls')            
        wb.save(xls_path)
        logger.info('Exporting excel success : {}'.format(xls_path))
        return xls_path   
    
    def make_android_html(self, scene, summary : dict, report_path=None):
        logger.info('Generating HTML ...')
        STATICPATH = os.path.dirname(os.path.realpath(__file__))
        file_loader = FileSystemLoader(os.path.join(STATICPATH, 'report_template'))
        env = Environment(loader=file_loader)
        template = env.get_template('android.html')
        if report_path:
            html_path = report_path
        else:
            html_path = os.path.join(self.report_dir, scene, 'report.html')   
        with open(html_path,'w+') as fout:
            html_content = template.render(devices=summary['devices'],app=summary['app'],
                                           platform=summary['platform'],ctime=summary['ctime'],
                                           cpu_app=summary['cpu_app'],cpu_sys=summary['cpu_sys'],
                                           mem_total=summary['mem_total'],mem_swap=summary['mem_swap'],
                                           fps=summary['fps'],jank=summary['jank'],level=summary['level'],
                                           tem=summary['tem'],net_send=summary['net_send'],
                                           net_recv=summary['net_recv'],cpu_charts=summary['cpu_charts'],
                                           mem_charts=summary['mem_charts'],net_charts=summary['net_charts'],
                                           battery_charts=summary['battery_charts'],fps_charts=summary['fps_charts'],
                                           jank_charts=summary['jank_charts'],mem_detail_charts=summary['mem_detail_charts'],
                                           gpu=summary['gpu'], gpu_charts=summary['gpu_charts'])
            
            fout.write(html_content)
        logger.info('Generating HTML success : {}'.format(html_path))  
        return html_path
    
    def make_ios_html(self, scene, summary : dict, report_path=None):
        logger.info('Generating HTML ...')
        STATICPATH = os.path.dirname(os.path.realpath(__file__))
        file_loader = FileSystemLoader(os.path.join(STATICPATH, 'report_template'))
        env = Environment(loader=file_loader)
        template = env.get_template('ios.html')
        if report_path:
            html_path = report_path
        else:
            html_path = os.path.join(self.report_dir, scene, 'report.html')
        with open(html_path,'w+') as fout:
            html_content = template.render(devices=summary['devices'],app=summary['app'],
                                           platform=summary['platform'],ctime=summary['ctime'],
                                           cpu_app=summary['cpu_app'],cpu_sys=summary['cpu_sys'],gpu=summary['gpu'],
                                           mem_total=summary['mem_total'],fps=summary['fps'],
                                           tem=summary['tem'],current=summary['current'],
                                           voltage=summary['voltage'],power=summary['power'],
                                           net_send=summary['net_send'],net_recv=summary['net_recv'],
                                           cpu_charts=summary['cpu_charts'],mem_charts=summary['mem_charts'],
                                           net_charts=summary['net_charts'],battery_charts=summary['battery_charts'],
                                           fps_charts=summary['fps_charts'],gpu_charts=summary['gpu_charts'])            
            fout.write(html_content)
        logger.info('Generating HTML success : {}'.format(html_path))  
        return html_path
  
    def filter_secen(self, scene):
        dirs = os.listdir(self.report_dir)
        dir_list = list(reversed(sorted(dirs, key=lambda x: os.path.getmtime(os.path.join(self.report_dir, x)))))
        dir_list.remove(scene)
        return dir_list

    def get_repordir(self):
        report_dir = os.path.join(os.getcwd(), 'report')
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        return report_dir

    def create_file(self, filename, content=''):
        if not os.path.exists(self.report_dir):
            os.mkdir(self.report_dir)
        with open(os.path.join(self.report_dir, filename), 'a+', encoding="utf-8") as file:
            file.write(content)

    def add_log(self, path, log_time, value):
        if value >= 0:
            with open(path, 'a+', encoding="utf-8") as file:
                file.write(f'{log_time}={str(value)}' + '\n')
    
    def record_net(self, type, send , recv):
        net_dict = dict()
        match(type):
            case 'pre':
                net_dict['send'] = send
                net_dict['recv'] = recv
                content = json.dumps(net_dict)
                self.create_file(filename='pre_net.json', content=content)
            case 'end':
                net_dict['send'] = send
                net_dict['recv'] = recv
                content = json.dumps(net_dict)
                self.create_file(filename='end_net.json', content=content)
            case _:
                logger.error('record network data failed')
    
    def make_report(self, app, devices, video, platform=Platform.Android, model='normal', cores=0):
        logger.info('Generating test results ...')
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        result_dict = {
            "app": app,
            "icon": "",
            "platform": platform,
            "model": model,
            "devices": devices,
            "ctime": current_time,
            "video": video,
            "cores":cores
        }
        content = json.dumps(result_dict)
        self.create_file(filename='result.json', content=content)
        report_new_dir = os.path.join(self.report_dir, f'apm_{current_time}')
        if not os.path.exists(report_new_dir):
            os.mkdir(report_new_dir)

        for f in os.listdir(self.report_dir):
            filename = os.path.join(self.report_dir, f)
            if f.split(".")[-1] in ['log', 'json', 'mkv']:
                shutil.move(filename, report_new_dir)        
        logger.info('Generating test results success: {}'.format(report_new_dir))
        return f'apm_{current_time}'        

    def instance_type(self, data):
        if isinstance(data, float):
            return 'float'
        elif isinstance(data, int):
            return 'int'
        else:
            return 'int'
    
    def open_file(self, path, mode):
        with open(path, mode) as f:
            for line in f:
                yield line
    
    def readJson(self, scene):
        path = os.path.join(self.report_dir,scene,'result.json')
        result_json = open(file=path, mode='r').read()
        result_dict = json.loads(result_json)
        return result_dict

    def readLog(self, scene, filename):
        """Read apmlog file data"""
        log_data_list = list()
        target_data_list = list()
        if os.path.exists(os.path.join(self.report_dir,scene,filename)):
            lines = self.open_file(os.path.join(self.report_dir,scene,filename), "r")
            for line in lines:
                if isinstance(line.split('=')[1].strip(), int):
                    log_data_list.append({
                        "x": line.split('=')[0].strip(),
                        "y": int(line.split('=')[1].strip())
                    })
                    target_data_list.append(int(line.split('=')[1].strip()))
                else:
                    log_data_list.append({
                        "x": line.split('=')[0].strip(),
                        "y": float(line.split('=')[1].strip())
                    })
                    target_data_list.append(float(line.split('=')[1].strip()))
        return log_data_list, target_data_list
        
    def getCpuLog(self, platform, scene):
        targetDic = dict()
        targetDic['cpuAppData'] = self.readLog(scene=scene, filename='cpu_app.log')[0]
        targetDic['cpuSysData'] = self.readLog(scene=scene, filename='cpu_sys.log')[0]
        result = {'status': 1, 'cpuAppData': targetDic['cpuAppData'], 'cpuSysData': targetDic['cpuSysData']}
        return result
    
    def getCpuLogCompare(self, platform, scene1, scene2):
        targetDic = dict()
        targetDic['scene1'] = self.readLog(scene=scene1, filename='cpu_app.log')[0]
        targetDic['scene2'] = self.readLog(scene=scene2, filename='cpu_app.log')[0]
        result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}
        return result
    
    def getGpuLog(self, platform, scene):
        targetDic = dict()
        targetDic['gpu'] = self.readLog(scene=scene, filename='gpu.log')[0]
        result = {'status': 1, 'gpu': targetDic['gpu']}
        return result
    
    def getGpuLogCompare(self, platform, scene1, scene2):
        targetDic = dict()
        targetDic['scene1'] = self.readLog(scene=scene1, filename='gpu.log')[0]
        targetDic['scene2'] = self.readLog(scene=scene2, filename='gpu.log')[0]
        result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}
        return result
    
    def getMemLog(self, platform, scene):
        targetDic = dict()
        targetDic['memTotalData'] = self.readLog(scene=scene, filename='mem_total.log')[0]
        if platform == Platform.Android:
            targetDic['memSwapData']  = self.readLog(scene=scene, filename='mem_swap.log')[0]
            result = {'status': 1, 
                      'memTotalData': targetDic['memTotalData'], 
                      'memSwapData': targetDic['memSwapData']}
        else:
            result = {'status': 1, 'memTotalData': targetDic['memTotalData']}
        return result
    
    def getMemDetailLog(self, platform, scene):
        targetDic = dict()
        targetDic['java_heap'] = self.readLog(scene=scene, filename='mem_java_heap.log')[0]
        targetDic['native_heap'] = self.readLog(scene=scene, filename='mem_native_heap.log')[0]
        targetDic['code_pss'] = self.readLog(scene=scene, filename='mem_code_pss.log')[0]
        targetDic['stack_pss'] = self.readLog(scene=scene, filename='mem_stack_pss.log')[0]
        targetDic['graphics_pss'] = self.readLog(scene=scene, filename='mem_graphics_pss.log')[0]
        targetDic['private_pss'] = self.readLog(scene=scene, filename='mem_private_pss.log')[0]
        targetDic['system_pss'] = self.readLog(scene=scene, filename='mem_system_pss.log')[0]
        result = {'status': 1, 'memory_detail': targetDic}
        return result
    
    def getCpuCoreLog(self, platform, scene):
        targetDic = dict()
        cores =self.readJson(scene=scene).get('cores', 0)
        if int(cores) > 0:
            for i in range(int(cores)):
                targetDic['cpu{}'.format(i)] = self.readLog(scene=scene, filename='cpu{}.log'.format(i))[0]
        result = {'status': 1, 'cores':cores, 'cpu_core': targetDic}
        return result
    
    def getMemLogCompare(self, platform, scene1, scene2):
        targetDic = dict()
        targetDic['scene1'] = self.readLog(scene=scene1, filename='mem_total.log')[0]
        targetDic['scene2'] = self.readLog(scene=scene2, filename='mem_total.log')[0]
        result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}
        return result
    
    def getBatteryLog(self, platform, scene):
        targetDic = dict()
        if platform == Platform.Android:
            targetDic['batteryLevel'] = self.readLog(scene=scene, filename='battery_level.log')[0]
            targetDic['batteryTem'] = self.readLog(scene=scene, filename='battery_tem.log')[0]
            result = {'status': 1, 
                      'batteryLevel': targetDic['batteryLevel'], 
                      'batteryTem': targetDic['batteryTem']}
        else:
            targetDic['batteryTem'] = self.readLog(scene=scene, filename='battery_tem.log')[0]
            targetDic['batteryCurrent'] = self.readLog(scene=scene, filename='battery_current.log')[0]
            targetDic['batteryVoltage'] = self.readLog(scene=scene, filename='battery_voltage.log')[0]
            targetDic['batteryPower'] = self.readLog(scene=scene, filename='battery_power.log')[0]
            result = {'status': 1, 
                      'batteryTem': targetDic['batteryTem'], 
                      'batteryCurrent': targetDic['batteryCurrent'],
                      'batteryVoltage': targetDic['batteryVoltage'], 
                      'batteryPower': targetDic['batteryPower']}    
        return result
    
    def getBatteryLogCompare(self, platform, scene1, scene2):
        targetDic = dict()
        if platform == Platform.Android:
            targetDic['scene1'] = self.readLog(scene=scene1, filename='battery_level.log')[0]
            targetDic['scene2'] = self.readLog(scene=scene2, filename='battery_level.log')[0]
            result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}
        else:
            targetDic['scene1'] = self.readLog(scene=scene1, filename='batteryPower.log')[0]
            targetDic['scene2'] = self.readLog(scene=scene2, filename='batteryPower.log')[0]
            result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}    
        return result
    
    def getFlowLog(self, platform, scene):
        targetDic = dict()
        targetDic['upFlow'] = self.readLog(scene=scene, filename='upflow.log')[0]
        targetDic['downFlow'] = self.readLog(scene=scene, filename='downflow.log')[0]
        result = {'status': 1, 'upFlow': targetDic['upFlow'], 'downFlow': targetDic['downFlow']}
        return result
    
    def getFlowSendLogCompare(self, platform, scene1, scene2):
        targetDic = dict()
        targetDic['scene1'] = self.readLog(scene=scene1, filename='upflow.log')[0]
        targetDic['scene2'] = self.readLog(scene=scene2, filename='upflow.log')[0]
        result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}
        return result
    
    def getFlowRecvLogCompare(self, platform, scene1, scene2):
        targetDic = dict()
        targetDic['scene1'] = self.readLog(scene=scene1, filename='downflow.log')[0]
        targetDic['scene2'] = self.readLog(scene=scene2, filename='downflow.log')[0]
        result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}
        return result
    
    def getFpsLog(self, platform, scene):
        targetDic = dict()
        targetDic['fps'] = self.readLog(scene=scene, filename='fps.log')[0]
        if platform == Platform.Android:
            targetDic['jank'] = self.readLog(scene=scene, filename='jank.log')[0]
            result = {'status': 1, 'fps': targetDic['fps'], 'jank': targetDic['jank']}
        else:
            result = {'status': 1, 'fps': targetDic['fps']}     
        return result
    
    def getDiskLog(self, platform, scene):
        targetDic = dict()
        targetDic['used'] = self.readLog(scene=scene, filename='disk_used.log')[0]
        targetDic['free'] = self.readLog(scene=scene, filename='disk_free.log')[0]
        result = {'status': 1, 'used': targetDic['used'], 'free':targetDic['free']}
        return result

    def analysisDisk(self, scene):
        initail_disk_list = list()
        current_disk_list = list()
        sum_init_disk = dict()
        sum_current_disk = dict()
        if os.path.exists(os.path.join(self.report_dir,scene,'initail_disk.log')):
            size_list = list()
            used_list = list()
            free_list = list()
            lines = self.open_file(os.path.join(self.report_dir,scene,'initail_disk.log'), "r")
            for line in lines:
                if 'Filesystem' not in line and line.strip() != '':
                    disk_value_list = line.split()
                    disk_dict = dict(
                        filesystem = disk_value_list[0],
                        blocks = disk_value_list[1],
                        used = disk_value_list[2],
                        available = disk_value_list[3],
                        use_percent = disk_value_list[4],
                        mounted = disk_value_list[5]
                    )
                    initail_disk_list.append(disk_dict)
                    size_list.append(int(disk_value_list[1]))
                    used_list.append(int(disk_value_list[2]))
                    free_list.append(int(disk_value_list[3]))
            sum_init_disk['sum_size'] = int(sum(size_list) / 1024 / 1024)
            sum_init_disk['sum_used'] = int(sum(used_list) / 1024)
            sum_init_disk['sum_free'] = int(sum(free_list) / 1024)
               
        if os.path.exists(os.path.join(self.report_dir,scene,'current_disk.log')):
            size_list = list()
            used_list = list()
            free_list = list()
            lines = self.open_file(os.path.join(self.report_dir,scene,'current_disk.log'), "r")
            for line in lines:
                if 'Filesystem' not in line and line.strip() != '':
                    disk_value_list = line.split()
                    disk_dict = dict(
                        filesystem = disk_value_list[0],
                        blocks = disk_value_list[1],
                        used = disk_value_list[2],
                        available = disk_value_list[3],
                        use_percent = disk_value_list[4],
                        mounted = disk_value_list[5]
                    )
                    current_disk_list.append(disk_dict)
                    size_list.append(int(disk_value_list[1]))
                    used_list.append(int(disk_value_list[2]))
                    free_list.append(int(disk_value_list[3]))
            sum_current_disk['sum_size'] = int(sum(size_list) / 1024 / 1024)
            sum_current_disk['sum_used'] = int(sum(used_list) / 1024)
            sum_current_disk['sum_free'] = int(sum(free_list) / 1024)       
                 
        return initail_disk_list, current_disk_list, sum_init_disk, sum_current_disk

    def getFpsLogCompare(self, platform, scene1, scene2):
        targetDic = dict()
        targetDic['scene1'] = self.readLog(scene=scene1, filename='fps.log')[0]
        targetDic['scene2'] = self.readLog(scene=scene2, filename='fps.log')[0]
        result = {'status': 1, 'scene1': targetDic['scene1'], 'scene2': targetDic['scene2']}
        return result
        
    def approximateSize(self, size, a_kilobyte_is_1024_bytes=True):
        '''
        convert a file size to human-readable form.
        Keyword arguments:
        size -- file size in bytes
        a_kilobyte_is_1024_bytes -- if True (default),use multiples of 1024
                                    if False, use multiples of 1000
        Returns: string
        '''

        suffixes = {1000: ['KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'],
                    1024: ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']}

        if size < 0:
            raise ValueError('number must be non-negative')

        multiple = 1024 if a_kilobyte_is_1024_bytes else 1000

        for suffix in suffixes[multiple]:
            size /= multiple
            if size < multiple:
                return '{0:.2f} {1}'.format(size, suffix)
    
    def _setAndroidPerfs(self, scene):
        """Aggregate APM data for Android"""
        
        app = self.readJson(scene=scene).get('app')
        devices = self.readJson(scene=scene).get('devices')
        platform = self.readJson(scene=scene).get('platform')
        ctime = self.readJson(scene=scene).get('ctime')

        cpuAppData = self.readLog(scene=scene, filename=f'cpu_app.log')[1]
        cpuSystemData = self.readLog(scene=scene, filename=f'cpu_sys.log')[1]
        if cpuAppData.__len__() > 0 and cpuSystemData.__len__() > 0:
            cpuAppRate = f'{round(sum(cpuAppData) / len(cpuAppData), 2)}%'
            cpuSystemRate = f'{round(sum(cpuSystemData) / len(cpuSystemData), 2)}%'
        else:
            cpuAppRate, cpuSystemRate = 0, 0    

        batteryLevelData = self.readLog(scene=scene, filename=f'battery_level.log')[1]
        batteryTemlData = self.readLog(scene=scene, filename=f'battery_tem.log')[1]
        if batteryLevelData.__len__() > 0 and batteryTemlData.__len__() > 0:
            batteryLevel = f'{batteryLevelData[-1]}%'
            batteryTeml = f'{batteryTemlData[-1]}°C'
        else:
            batteryLevel, batteryTeml = 0, 0   
    

        totalPassData = self.readLog(scene=scene, filename=f'mem_total.log')[1]
        
        if totalPassData.__len__() > 0:
            swapPassData = self.readLog(scene=scene, filename=f'mem_swap.log')[1]
            totalPassAvg = f'{round(sum(totalPassData) / len(totalPassData), 2)}MB'
            swapPassAvg = f'{round(sum(swapPassData) / len(swapPassData), 2)}MB'
        else:
            totalPassAvg, swapPassAvg = 0, 0    

        fpsData = self.readLog(scene=scene, filename=f'fps.log')[1]
        jankData = self.readLog(scene=scene, filename=f'jank.log')[1]
        if fpsData.__len__() > 0:
            fpsAvg = f'{int(sum(fpsData) / len(fpsData))}HZ/s'
            jankAvg = f'{int(sum(jankData))}'
        else:
            fpsAvg, jankAvg = 0, 0    

        if os.path.exists(os.path.join(self.report_dir,scene,'end_net.json')):
            f_pre = open(os.path.join(self.report_dir,scene,'pre_net.json'))
            f_end = open(os.path.join(self.report_dir,scene,'end_net.json'))
            json_pre = json.loads(f_pre.read())
            json_end = json.loads(f_end.read())
            send = json_end['send'] - json_pre['send']
            recv = json_end['recv'] - json_pre['recv']
        else:
            send, recv = 0, 0    
        flowSend = f'{round(float(send / 1024), 2)}MB'
        flowRecv = f'{round(float(recv / 1024), 2)}MB'

        gpuData = self.readLog(scene=scene, filename='gpu.log')[1]
        if gpuData.__len__() > 0:
            gpu = round(sum(gpuData) / len(gpuData), 2)
        else:
            gpu = 0

        mem_detail_flag = os.path.exists(os.path.join(self.report_dir,scene,'mem_java_heap.log'))
        disk_flag = os.path.exists(os.path.join(self.report_dir,scene,'disk_free.log'))
        thermal_flag = os.path.exists(os.path.join(self.report_dir,scene,'init_thermal_temp.json'))
        cpu_core_flag = os.path.exists(os.path.join(self.report_dir,scene,'cpu0.log'))
        apm_dict = dict()
        apm_dict['app'] = app
        apm_dict['devices'] = devices
        apm_dict['platform'] = platform
        apm_dict['ctime'] = ctime
        apm_dict['cpuAppRate'] = cpuAppRate
        apm_dict['cpuSystemRate'] = cpuSystemRate
        apm_dict['totalPassAvg'] = totalPassAvg
        apm_dict['swapPassAvg'] = swapPassAvg
        apm_dict['fps'] = fpsAvg
        apm_dict['jank'] = jankAvg
        apm_dict['flow_send'] = flowSend
        apm_dict['flow_recv'] = flowRecv
        apm_dict['batteryLevel'] = batteryLevel
        apm_dict['batteryTeml'] = batteryTeml
        apm_dict['mem_detail_flag'] = mem_detail_flag
        apm_dict['disk_flag'] = disk_flag
        apm_dict['gpu'] = gpu
        apm_dict['thermal_flag'] = thermal_flag
        apm_dict['cpu_core_flag'] = cpu_core_flag
        
        if thermal_flag:
            init_thermal_temp = json.loads(open(os.path.join(self.report_dir,scene,'init_thermal_temp.json')).read())
            current_thermal_temp = json.loads(open(os.path.join(self.report_dir,scene,'current_thermal_temp.json')).read())
            apm_dict['init_thermal_temp'] = init_thermal_temp
            apm_dict['current_thermal_temp'] = current_thermal_temp

        return apm_dict

    def _setiOSPerfs(self, scene):
        """Aggregate APM data for iOS"""
        
        app = self.readJson(scene=scene).get('app')
        devices = self.readJson(scene=scene).get('devices')
        platform = self.readJson(scene=scene).get('platform')
        ctime = self.readJson(scene=scene).get('ctime')

        cpuAppData = self.readLog(scene=scene, filename=f'cpu_app.log')[1]
        cpuSystemData = self.readLog(scene=scene, filename=f'cpu_sys.log')[1]
        if cpuAppData.__len__() > 0 and cpuSystemData.__len__() > 0:
            cpuAppRate = f'{round(sum(cpuAppData) / len(cpuAppData), 2)}%'
            cpuSystemRate = f'{round(sum(cpuSystemData) / len(cpuSystemData), 2)}%'
        else:
            cpuAppRate, cpuSystemRate = 0, 0

        totalPassData = self.readLog(scene=scene, filename='mem_total.log')[1]
        if totalPassData.__len__() > 0:
            totalPassAvg = f'{round(sum(totalPassData) / len(totalPassData), 2)}MB'
        else:
            totalPassAvg = 0

        fpsData = self.readLog(scene=scene, filename='fps.log')[1]
        if fpsData.__len__() > 0:
            fpsAvg = f'{int(sum(fpsData) / len(fpsData))}HZ/s'
        else:
            fpsAvg = 0

        flowSendData = self.readLog(scene=scene, filename='upflow.log')[1]
        flowRecvData = self.readLog(scene=scene, filename='downflow.log')[1]
        if flowSendData.__len__() > 0:
            flowSend = f'{round(float(sum(flowSendData) / 1024), 2)}MB'
            flowRecv = f'{round(float(sum(flowRecvData) / 1024), 2)}MB'
        else:
            flowSend, flowRecv = 0, 0    

        batteryTemlData = self.readLog(scene=scene, filename='battery_tem.log')[1]
        batteryCurrentData = self.readLog(scene=scene, filename='battery_current.log')[1]
        batteryVoltageData = self.readLog(scene=scene, filename='battery_voltage.log')[1]
        batteryPowerData = self.readLog(scene=scene, filename='battery_power.log')[1]
        if batteryTemlData.__len__() > 0:
            batteryTeml = int(batteryTemlData[-1])
            batteryCurrent = int(sum(batteryCurrentData) / len(batteryCurrentData))
            batteryVoltage = int(sum(batteryVoltageData) / len(batteryVoltageData))
            batteryPower = int(sum(batteryPowerData) / len(batteryPowerData))
        else:
            batteryTeml,  batteryCurrent , batteryVoltage, batteryPower = 0, 0, 0, 0 

        gpuData = self.readLog(scene=scene, filename='gpu.log')[1]
        if gpuData.__len__() > 0:
            gpu = round(sum(gpuData) / len(gpuData), 2)
        else:
            gpu = 0    
        disk_flag = os.path.exists(os.path.join(self.report_dir,scene,'disk_free.log'))
        apm_dict = dict()
        apm_dict['app'] = app
        apm_dict['devices'] = devices
        apm_dict['platform'] = platform
        apm_dict['ctime'] = ctime
        apm_dict['cpuAppRate'] = cpuAppRate
        apm_dict['cpuSystemRate'] = cpuSystemRate
        apm_dict['totalPassAvg'] = totalPassAvg
        apm_dict['nativePassAvg'] = 0
        apm_dict['dalvikPassAvg'] = 0
        apm_dict['fps'] = fpsAvg
        apm_dict['jank'] = 0
        apm_dict['flow_send'] = flowSend
        apm_dict['flow_recv'] = flowRecv
        apm_dict['batteryTeml'] = batteryTeml
        apm_dict['batteryCurrent'] = batteryCurrent
        apm_dict['batteryVoltage'] = batteryVoltage
        apm_dict['batteryPower'] = batteryPower
        apm_dict['gpu'] = gpu
        apm_dict['disk_flag'] = disk_flag
        return apm_dict

    def _setpkPerfs(self, scene):
        """Aggregate APM data for pk model"""
        cpuAppData1 = self.readLog(scene=scene, filename='cpu_app1.log')[1]
        cpuAppRate1 = f'{round(sum(cpuAppData1) / len(cpuAppData1), 2)}%'
        cpuAppData2 = self.readLog(scene=scene, filename='cpu_app2.log')[1]
        cpuAppRate2 = f'{round(sum(cpuAppData2) / len(cpuAppData2), 2)}%'

        totalPassData1 = self.readLog(scene=scene, filename='mem1.log')[1]
        totalPassAvg1 = f'{round(sum(totalPassData1) / len(totalPassData1), 2)}MB'
        totalPassData2 = self.readLog(scene=scene, filename='mem2.log')[1]
        totalPassAvg2 = f'{round(sum(totalPassData2) / len(totalPassData2), 2)}MB'

        fpsData1 = self.readLog(scene=scene, filename='fps1.log')[1]
        fpsAvg1 = f'{int(sum(fpsData1) / len(fpsData1))}HZ/s'
        fpsData2 = self.readLog(scene=scene, filename='fps2.log')[1]
        fpsAvg2 = f'{int(sum(fpsData2) / len(fpsData2))}HZ/s'

        networkData1 = self.readLog(scene=scene, filename='network1.log')[1]
        network1 = f'{round(float(sum(networkData1) / 1024), 2)}MB'
        networkData2 = self.readLog(scene=scene, filename='network2.log')[1]
        network2 = f'{round(float(sum(networkData2) / 1024), 2)}MB'
        
        apm_dict = dict()
        apm_dict['cpuAppRate1'] = cpuAppRate1
        apm_dict['cpuAppRate2'] = cpuAppRate2
        apm_dict['totalPassAvg1'] = totalPassAvg1
        apm_dict['totalPassAvg2'] = totalPassAvg2
        apm_dict['network1'] = network1
        apm_dict['network2'] = network2
        apm_dict['fpsAvg1'] = fpsAvg1
        apm_dict['fpsAvg2'] = fpsAvg2
        return apm_dict

class Method:
    
    @classmethod
    def _request(cls, request, object):
        match(request.method):
            case 'POST':
                return request.form[object]
            case 'GET':
                return request.args[object]
            case _:
                raise Exception('request method error')
    
    @classmethod   
    def _setValue(cls, value, default = 0):
        try:
            result = value
        except ZeroDivisionError :
            result = default
        except IndexError:
            result = default        
        except Exception:
            result = default            
        return result
    
    @classmethod
    def _settings(cls, request):
        content = {}
        content['cpuWarning'] = (0, request.cookies.get('cpuWarning'))[request.cookies.get('cpuWarning') not in [None, 'NaN']]
        content['memWarning'] = (0, request.cookies.get('memWarning'))[request.cookies.get('memWarning') not in [None, 'NaN']]
        content['fpsWarning'] = (0, request.cookies.get('fpsWarning'))[request.cookies.get('fpsWarning') not in [None, 'NaN']]
        content['netdataRecvWarning'] = (0, request.cookies.get('netdataRecvWarning'))[request.cookies.get('netdataRecvWarning') not in [None, 'NaN']]
        content['netdataSendWarning'] = (0, request.cookies.get('netdataSendWarning'))[request.cookies.get('netdataSendWarning') not in [None, 'NaN']]
        content['betteryWarning'] = (0, request.cookies.get('betteryWarning'))[request.cookies.get('betteryWarning') not in [None, 'NaN']]
        content['gpuWarning'] = (0, request.cookies.get('gpuWarning'))[request.cookies.get('gpuWarning') not in [None, 'NaN']]
        content['duration'] = (0, request.cookies.get('duration'))[request.cookies.get('duration') not in [None, 'NaN']]
        content['solox_host'] = ('', request.cookies.get('solox_host'))[request.cookies.get('solox_host') not in [None, 'NaN']]
        content['host_switch'] = request.cookies.get('host_switch')
        return content
    
    @classmethod
    def _index(cls, target: list, index: int, default: any):
        try:
            return target[index]
        except IndexError:
            return default

class Install:

    def uploadFile(self, file_path, file_obj):
        """save upload file"""
        try:
            file_obj.save(file_path)
            return True
        except Exception as e:
            logger.exception(e)
            return False            

    def downloadLink(self,filelink=None, path=None, name=None):
        try:
            logger.info('Install link : {}'.format(filelink))
            ssl._create_default_https_context = ssl._create_unverified_context
            file_size = int(urlopen(filelink).info().get('Content-Length', -1))
            header = {"Range": "bytes=%s-%s" % (0, file_size)}
            pbar = tqdm(
                total=file_size, initial=0,
                unit='B', unit_scale=True, desc=filelink.split('/')[-1])
            req = requests.get(filelink, headers=header, stream=True)
            with(open(os.path.join(path, name), 'ab')) as f:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                         f.write(chunk)
                         pbar.update(1024)
            pbar.close()
            return True
        except Exception as e:
            logger.exception(e)
            return False

    def installAPK(self, path):
        result = adb.shell_noDevice(cmd='install -r {}'.format(path))
        if result == 0:
            os.remove(path)
            return True, result
        else:
            return False, result

    def installIPA(self, path):
        result = os.system('tidevice install {}'.format(path))
        if result == 0:
            os.remove(path)
            return True, result
        else:
            return False, result

class Scrcpy:

    STATICPATH = os.path.dirname(os.path.realpath(__file__))
    DEFAULT_SCRCPY_PATH = {
        "64": os.path.join(STATICPATH, "scrcpy", "scrcpy-win64-v2.4", "scrcpy.exe"),
        "32": os.path.join(STATICPATH, "scrcpy", "scrcpy-win32-v2.4", "scrcpy.exe"),
        "default":"scrcpy"
    }
    
    @classmethod
    def scrcpy_path(cls):
        bit = platform.architecture()[0]
        path = cls.DEFAULT_SCRCPY_PATH["default"]
        if platform.system().lower().__contains__('windows'):
            if bit.__contains__('64'):
                path =  cls.DEFAULT_SCRCPY_PATH["64"]
            elif bit.__contains__('32'):
                path =  cls.DEFAULT_SCRCPY_PATH["32"]
        return path
    
    @classmethod
    def start_record(cls, device):
        f = File()
        logger.info('start record screen')
        win_cmd = "start /b {scrcpy_path} -s {deviceId} --no-playback --record={video}".format(
            scrcpy_path = cls.scrcpy_path(), 
            deviceId = device, 
            video = os.path.join(f.report_dir, 'record.mkv')
        )
        mac_cmd = "nohup {scrcpy_path} -s {deviceId} --no-playback --record={video} &".format(
            scrcpy_path = cls.scrcpy_path(), 
            deviceId = device, 
            video = os.path.join(f.report_dir, 'record.mkv')
        )
        if platform.system().lower().__contains__('windows'):
            result = os.system(win_cmd)
        else:
            result = os.system(mac_cmd)    
        if result == 0:
            logger.info("record screen success : {}".format(os.path.join(f.report_dir, 'record.mkv')))
        else:
            logger.error("solox's scrcpy is incompatible with your PC")
            logger.info("Please install the software yourself : brew install scrcpy")    
        return result
    
    @classmethod
    def stop_record(cls):
        logger.info('stop scrcpy process')
        pids = psutil.pids()
        try:
            for pid in pids:
                try:
                    p = psutil.Process(pid)
                    if p.name().__contains__('scrcpy'):
                        os.kill(pid, signal.SIGABRT)
                        logger.info(pid)
                except psutil.ZombieProcess:
                    logger.warning(f"Skipped zombie process with PID: {pid}")
                except psutil.AccessDenied:
                    logger.warning(f"Access denied to process with PID: {pid}")
                except psutil.NoSuchProcess:
                    logger.warning(f"Process no longer exists with PID: {pid}")
        except Exception as e:
            logger.exception(e)
    
    @classmethod
    def cast_screen(cls, device):
        logger.info('start cast screen')
        win_cmd = "start /i {scrcpy_path} -s {deviceId} --stay-awake".format(
            scrcpy_path = cls.scrcpy_path(), 
            deviceId = device
        )
        mac_cmd = "nohup {scrcpy_path} -s {deviceId} --stay-awake &".format(
            scrcpy_path = cls.scrcpy_path(), 
            deviceId = device
        )
        if platform.system().lower().__contains__('windows'):
            result = os.system(win_cmd)
        else:
            result = os.system(mac_cmd)
        if result == 0:
            logger.info("cast screen success")
        else:
            logger.error("solox's scrcpy is incompatible with your PC")
            logger.info("Please install the software yourself : brew install scrcpy")    
        return result
    
    @classmethod
    def play_video(cls, video):
        logger.info('start play video : {}'.format(video))
        cap = cv2.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.namedWindow("frame", 0)  
                cv2.resizeWindow("frame", 430, 900)
                cv2.imshow('frame', gray)
                if cv2.waitKey(25) & 0xFF == ord('q') or not cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
