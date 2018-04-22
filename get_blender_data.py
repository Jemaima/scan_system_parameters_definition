# obj = bpy.context.active_object
# obj.location #for transform_vector
# for v in obj.data.vertices:
#     print(v.co)
# inspect.getmembers(obj, lambda a:not(inspect.isroutine(a)))  #to find all atribules of class
# obj.rotation_euler # .x, .y, .z * 180/pi to conver to angled


import bpy
from math import pi
obj = bpy.context.active_object

f = open('C:\\Users\\metel\\PycharmProjects\\PnP_problem\\object.txt', 'w', newline="\n")
f.write(','.join([str(obj.location.x),str(obj.location.y),str(obj.location.z)]))
f.write('\n')
f.write(','.join([str(obj.rotation_euler.x*180/pi), str(obj.rotation_euler.y*180/pi), str(obj.rotation_euler.z*180/pi)]))
f.write('\n')
f.write(str(len(obj.data.vertices)))
f.write('\n')
for v in obj.data.vertices:
    f.write(','.join([str(v.co.x), str(v.co.y), str(v.co.z)]))
    f.write('\n')
f.close()
