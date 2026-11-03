# Lists playback devices or changes the Windows default playback device.
#   audio_default.ps1 list          -> one line per device: id|friendly name|default(0/1)
#   audio_default.ps1 set <id>      -> make <id> the default for all roles
param([string]$Action = "list", [string]$Id = "")

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

[ComImport, Guid("f8679f50-850a-41cf-9c72-430f290290c8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPolicyConfig {
    int GetMixFormat(); int GetDeviceFormat(); int ResetDeviceFormat(); int SetDeviceFormat();
    int GetProcessingPeriod(); int SetProcessingPeriod(); int GetShareMode(); int SetShareMode();
    int GetPropertyValue(); int SetPropertyValue();
    [PreserveSig] int SetDefaultEndpoint([MarshalAs(UnmanagedType.LPWStr)] string id, int role);
}
[ComImport, Guid("870af99c-171d-4f9e-af0d-e63df40c2bc9")] class CPolicyConfigClient {}

[ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    int Activate(); int OpenPropertyStore();
    [PreserveSig] int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
}
[ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int EnumAudioEndpoints();
    [PreserveSig] int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice dev);
}
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator {}

public static class AudioDefault {
    public static string Get() {
        var e = (IMMDeviceEnumerator)new MMDeviceEnumerator();
        IMMDevice dev; string id;
        if (e.GetDefaultAudioEndpoint(0, 1, out dev) != 0) return "";
        dev.GetId(out id);
        return id;
    }
    public static void Set(string id) {
        var p = (IPolicyConfig)new CPolicyConfigClient();
        for (int role = 0; role < 3; role++) Marshal.ThrowExceptionForHR(p.SetDefaultEndpoint(id, role));
    }
}
"@

if ($Action -eq "set") { [AudioDefault]::Set($Id); exit 0 }

$current = [AudioDefault]::Get()
$root = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render"
foreach ($key in Get-ChildItem $root) {
    if ((Get-ItemProperty $key.PSPath).DeviceState -ne 1) { continue }  # 1 = active
    $props = Get-ItemProperty (Join-Path $key.PSPath "Properties")
    $name = "$($props.'{a45c254e-df1c-4efd-8020-67d146a850e0},2') ($($props.'{b3f8fa53-0004-438e-9003-51a46e139bfc},6'))"
    $id = "{0.0.0.00000000}.$($key.PSChildName)"
    "$id|$name|$([int]($id -eq $current))"
}
