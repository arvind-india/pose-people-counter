import PySpin

system = PySpin.System.GetInstance()

cam_list = system.GetCameras()

camera = cam_list.GetByIndex(0)

camera.Init()

camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

camera.BeginAcquisition()

image_primary = camera.GetNextImage()

image_primary.Save('test_img_1.png')

image_primary = camera.GetNextImage()

image_primary.Save('test_img_2.png')

image_primary = camera.GetNextImage()

image_primary.Save('test_img_3.png')

cam_list.Clear()
system.ReleaseInstance()
