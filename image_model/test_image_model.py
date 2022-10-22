import os

folder_list = os.listdir("./data/fashion/251_class_train")
    
class_list = [len(os.listdir("./data/fashion/251_class_train/" + folder_i)) for folder_i in folder_list]
class_list.sort()

print(class_list)



folder_list = os.listdir("./data/fashion/251_class_test")
    
class_list = [len(os.listdir("./data/fashion/251_class_test/" + folder_i)) for folder_i in folder_list]
class_list.sort()

print(class_list)