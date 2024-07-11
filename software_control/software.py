import pyautogui
import numpy as np
import pandas as pd
import logging
import os
from time import sleep
from datetime import datetime
from ml import utils


class Software:
    def __init__(self, icons_dir, *args, **kwargs):

        # logger settings
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('MOR_RL.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # pyautogui settings
        self.confidence = 0.9
        self.confidence_high = 0.99
        self.confidence_low = 0.6
        pyautogui.FAILSAFE = False

        # directory settings
        self.icons_dir = icons_dir

        # variable settings
        self.running = None

    @staticmethod
    def get_cur_time(format='log'):
        assert format in ['log', 'file_name', 'sql']
        cur_time = datetime.now()
        if format == 'log':
            return cur_time.strftime('%Y-%m-%d %H:%M:%S')
        elif format == 'file_name':
            return cur_time.strftime('%Y-%m-%d_%H-%M-%S')
        else:
            return cur_time.replace(microsecond=0)

    def pass_screen_grab_error(func):
        def error_passer(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except OSError as error:
                print(f'{error}')
                return None
        return error_passer

    @pass_screen_grab_error
    def move_to_icon(self, file_path, confidence=None, **kwargs):
        confidence = confidence or self.confidence
        # if no icon detected on screen, will return None
        file_full_path = f'{self.icons_dir}/{file_path}'
        return pyautogui.locateCenterOnScreen(file_full_path, confidence=confidence)

    def click_button(self, file_path, **kwargs):
        sleep(0.5)
        t = 0
        while t < 100:
            try:
                if isinstance(file_path, str):
                    x, y = self.move_to_icon(file_path, **kwargs)
                    pyautogui.click(x, y)
                elif isinstance(file_path, list):
                    for icon in file_path:
                        if self.move_to_icon(icon, **kwargs):
                            x, y = self.move_to_icon(icon, **kwargs)
                            pyautogui.click(x, y)
                return None
            except TypeError:
                self.logger.warning(f'cannot find {file_path} on screen')
                t += 1
                sleep(3)
        self.logger.error(f'5 min timed out for finding {file_path}')
        raise SystemError(f'5 min timed out for finding {file_path}')

    def click_button_and_check_change(self, file_path, changed_file_path=None, exist=False):
        """
        this function is used to make sure the button is indeed clicked, by checking whether an icon
        exists (or not) on the screen. The default logic is that after we click a button, it will
        disappear.
        :param file_path: button to click
        :param changed_file_path: icon to check
        :param exist: icon expected to show (True) or not show (False) on the screen after click the button
        :return: None
        """
        changed_file_path = changed_file_path or file_path
        while True:
            self.click_button(file_path)
            sleep(1)
            if exist:
                if self.move_to_icon(changed_file_path):
                    break
            else:
                if not self.move_to_icon(changed_file_path):
                    break

    def right_click_and_check_change(self, changed_file_path):
        """
        this function is used to make sure right click successfully proceeded, by checking whether an icon
        exists on the screen.
        :param changed_file_path: icon to check
        :return: None
        """
        while True:
            pyautogui.rightClick()
            sleep(1)
            if self.move_to_icon(changed_file_path):
                break

    def bypass_teamviewer_prompt_window(self):
        if self.move_to_icon('teamviewer_window.png', confidence=self.confidence) or \
                self.move_to_icon('teamviewer_window_grey.png', confidence=self.confidence) or \
                self.move_to_icon('teamviewer.png', confidence=self.confidence):
            self.click_button_and_check_change('teamviewer_ok.png')
            self.logger.warning(f'teamviewer prompt window bypassed!')

    # check whether software is available on the screen
    def check_in_software(self):
        if self.move_to_icon('ec_lab_title.png', confidence=self.confidence):
            return True
        else:
            return False

    # open software if not in software interface, else do nothing
    def open_software(self):
        i = 0
        while not self.check_in_software() and i < 3:
            self.click_button_and_check_change('ec_lab_icon.png', 'ec_lab_title.png', exist=True)
            i += 1
            self.logger.info(f'attempting to open software, trial {i} in 3 times...')
            sleep(0.5)

    def get_act_name(self, act_id):
        return f'{act_id}_{self.get_cur_time("file_name")}'

    def start_exp(self, act_name, action):
        if self.running:
            self.logger.warning('Experiment is already running!')
            raise SystemError('Experiment is already running! Restart needed!')
        else:
            # start the experiment
            self.open_software()
            self.modify_protocol(action)
            self.click_button_and_check_change('ec_lab_start.png', 'ec_lab_save.png', exist=True)
            sleep(0.5)
            pyautogui.write(act_name, interval=0.1)
            sleep(1)
            self.click_button_and_check_change('ec_lab_save.png', exist=False)
            self.running = True

    def stop_exp(self):
        if self.running:
            self.open_software()
            self.click_button('ec_lab_stop_button.png')
            self.running = False
        else:
            self.logger.warning('Experiment is not running!')

    def check_exp_finished(self):
        while self.running:
            # bypass teamviewer window if exist
            self.bypass_teamviewer_prompt_window()
            # if not in software, go back to software if no human movement
            x, y = pyautogui.position()
            sleep(10)

            # check if human is operating PC, any cursor movement during 10s
            if (x, y) == pyautogui.position():
                # return to software if not in software
                if not self.check_in_software():
                    self.open_software()
                start_button_location = self.move_to_icon('ec_lab_start.png', confidence=self.confidence)
                if start_button_location is not None:
                    self.running = False
            else:
                self.logger.warning(f'human operating, fail to detect...')

    @staticmethod
    def move_to_screen_center():
        screenWidth, screenHeight = pyautogui.size()
        pyautogui.moveTo(screenWidth / 2, screenHeight / 2)

    def update_input(self, file_path, new_input):
        # select input area
        self.click_button(file_path, confidence=self.confidence_low)
        sleep(0.5)
        # clear previous input
        self.right_click_and_check_change('ec_lab_select_all.png')
        self.click_button_and_check_change('ec_lab_select_all.png')
        sleep(0.5)
        # input new parameter
        pyautogui.write(str(new_input), interval=0.1)
        sleep(0.5)

    def change_ca_settings(self, step_icon_list, potential, time):
        # convert time to min and sec
        min, sec = divmod(time, 60)

        # update rest step, and check if it turned blue
        self.click_button_and_check_change(step_icon_list, step_icon_list[1], exist=True)

        # update potential
        self.update_input('ec_lab_CA_potential.png', potential)

        # update time
        self.update_input('ec_lab_CA_min.png', int(min))
        self.update_input('ec_lab_CA_sec.png', int(sec))

    def change_loop_settings(self, loop_num):
        # update loop number, should be loop_num -1, because when input 0, run 1 time in default
        self.update_input('ec_lab_CA_loop.png', loop_num - 1)

    def modify_protocol(self, converted_action):
        potential_rest, potential_work, time_rest, time_work = converted_action
        # icon list settings
        ca_icon_list = [f'ec_lab_{icon}.png' for icon in ['CA', 'CA_blue', 'CA_grey']]
        ca_step_0_list = [f'ec_lab_{icon}.png' for icon in ['CA_step_0', 'CA_step_0_blue']]
        ca_step_1_list = [f'ec_lab_{icon}.png' for icon in ['CA_step_1', 'CA_step_1_blue']]

        # select CA technique
        self.click_button_and_check_change(ca_icon_list, 'ec_lab_CA_blue.png', exist=True)

        # enter modify mode
        if not self.move_to_icon('ec_lab_modify_in.png', confidence=self.confidence):
            self.click_button_and_check_change('ec_lab_modify.png', 'ec_lab_modify_in.png', exist=True)

        # update rest protocol
        self.change_ca_settings(ca_step_0_list, potential_rest, time_rest)

        # update loop times
        self.change_loop_settings(1)

        # update work protocol
        self.change_ca_settings(ca_step_1_list, potential_work, time_work)

        # update loop times
        self.change_loop_settings(utils.get_step_actual_time(converted_action, converted=True)[1])

    # export eis raw data to mpt file
    def export_to_txt(self, file_name):
        self.open_software()
        self.click_button_and_check_change('ec_lab_experiment.png', 'ec_lab_export_as_text.png', exist=True)
        self.click_button_and_check_change('ec_lab_export_as_text.png', 'ec_lab_add.png', exist=True)
        self.click_button('ec_lab_add.png')
        sleep(0.5)
        pyautogui.write(file_name, interval=0.1)
        sleep(0.5)
        pyautogui.press('enter')
        sleep(0.5)
        pyautogui.hotkey('alt', 'e')
        sleep(0.5)
        pyautogui.hotkey('alt', 'c')
