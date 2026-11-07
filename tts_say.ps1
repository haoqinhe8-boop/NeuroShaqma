# 小沙嘴巴（TTS）：Windows SAPI 文本 → WAV 文件（零依赖真实发声）
# 用法: powershell -File tts_say.ps1 -Text "..." -OutWav "D:\xx.wav"
param(
  [Parameter(Mandatory=$true)][string]$Text,
  [Parameter(Mandatory=$true)][string]$OutWav
)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile($OutWav)
$s.Speak($Text)
$s.Dispose()
Write-Output "TTS_OK|$OutWav"
