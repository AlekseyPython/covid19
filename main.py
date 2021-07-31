import Settings, Initialization


if __name__ == '__main__':
    if Settings.console_mode:
        from Business.TaskConvertDataToDesiredTypes import Task
        
        task = Task()
        task.start()
        task.join()
    else:
        ipresentation = Initialization.ipresentation
        ipresentation.build_mian_window()