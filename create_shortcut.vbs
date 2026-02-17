Set WshShell = WScript.CreateObject("WScript.Shell")
Set lnk = WshShell.CreateShortcut("C:\Users\conno\Desktop\Meridian.lnk")
lnk.TargetPath = "C:\Users\conno\OneDrive\Documents\Portfolio Optimization\meridian.bat"
lnk.WorkingDirectory = "C:\Users\conno\OneDrive\Documents\Portfolio Optimization"
lnk.IconLocation = "C:\Users\conno\OneDrive\Documents\Portfolio Optimization\src\portopt\assets\icon.ico"
lnk.WindowStyle = 7
lnk.Description = "Meridian Portfolio Terminal"
lnk.Save
