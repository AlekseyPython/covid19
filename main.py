import Settings, Initialization
import warnings


warnings.simplefilter(action='ignore')

if __name__ == '__main__':
    if Settings.console_mode:
        from Business.ConvertDataToDesiredTypes import Task
        
        task = Task()
        task.start()
        task.join()
    else:
        ipresentation = Initialization.ipresentation
        ipresentation.build_mian_window()