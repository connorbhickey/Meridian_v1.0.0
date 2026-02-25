; Meridian Portfolio Terminal — NSIS Installer Script
; Builds a proper Windows installer from PyInstaller output

;---------------------------------------------------------------------------
; Includes
;---------------------------------------------------------------------------
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

;---------------------------------------------------------------------------
; Defines
;---------------------------------------------------------------------------
!define PRODUCT_NAME "Meridian"
!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "1.0.2"
!endif
!define PRODUCT_PUBLISHER "Meridian"
!define PRODUCT_WEB_SITE "https://github.com/connorbhickey/Meridian_v1.0.0"
!define PRODUCT_DIR_REGKEY "Software\${PRODUCT_NAME}"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

; Source directory — PyInstaller output
!define DIST_DIR "..\..\dist\Meridian"

;---------------------------------------------------------------------------
; General
;---------------------------------------------------------------------------
Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "..\..\dist\Meridian-Setup.exe"
InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""
ShowInstDetails show
ShowUnInstDetails show
RequestExecutionLevel admin
SetCompressor /SOLID lzma
SetCompressorDictSize 64

;---------------------------------------------------------------------------
; MUI Settings
;---------------------------------------------------------------------------
!define MUI_ABORTWARNING
!define MUI_ICON "..\..\src\portopt\assets\icon.ico"
!define MUI_UNICON "..\..\src\portopt\assets\icon.ico"
!define MUI_WELCOMEPAGE_TITLE "Welcome to ${PRODUCT_NAME} Setup"
!define MUI_WELCOMEPAGE_TEXT "This wizard will install ${PRODUCT_NAME} ${PRODUCT_VERSION} on your computer.$\r$\n$\r$\n${PRODUCT_NAME} is a professional portfolio optimization terminal with real-time data, backtesting, and AI-powered analysis.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN "$INSTDIR\Meridian.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch ${PRODUCT_NAME}"

;---------------------------------------------------------------------------
; Pages
;---------------------------------------------------------------------------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;---------------------------------------------------------------------------
; Language
;---------------------------------------------------------------------------
!insertmacro MUI_LANGUAGE "English"

;---------------------------------------------------------------------------
; Version Info
;---------------------------------------------------------------------------
VIProductVersion "${PRODUCT_VERSION}.0"
VIAddVersionKey /LANG=${LANG_ENGLISH} "ProductName" "${PRODUCT_NAME}"
VIAddVersionKey /LANG=${LANG_ENGLISH} "ProductVersion" "${PRODUCT_VERSION}"
VIAddVersionKey /LANG=${LANG_ENGLISH} "CompanyName" "${PRODUCT_PUBLISHER}"
VIAddVersionKey /LANG=${LANG_ENGLISH} "FileDescription" "${PRODUCT_NAME} Portfolio Terminal Installer"
VIAddVersionKey /LANG=${LANG_ENGLISH} "FileVersion" "${PRODUCT_VERSION}"

;---------------------------------------------------------------------------
; Install Section
;---------------------------------------------------------------------------
Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    SetOverwrite on

    ; Copy the entire PyInstaller output
    File /r "${DIST_DIR}\*.*"

    ; Create Start Menu shortcuts
    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\Meridian.exe" "" "$INSTDIR\Meridian.exe" 0
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\uninst.exe"

    ; Create Desktop shortcut
    CreateShortCut "$DESKTOP\${PRODUCT_NAME}.lnk" "$INSTDIR\Meridian.exe" "" "$INSTDIR\Meridian.exe" 0

    ; Write registry keys
    WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\Meridian.exe"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninst.exe"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\Meridian.exe"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
    WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "NoModify" 1
    WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "NoRepair" 1

    ; Calculate installed size
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "EstimatedSize" $0

    ; Write uninstaller
    WriteUninstaller "$INSTDIR\uninst.exe"
SectionEnd

;---------------------------------------------------------------------------
; Uninstall Section
;---------------------------------------------------------------------------
Section Uninstall
    ; Kill app if running
    nsExec::ExecToLog 'taskkill /F /IM Meridian.exe'

    ; Remove Start Menu shortcuts
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk"
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk"
    RMDir "$SMPROGRAMS\${PRODUCT_NAME}"

    ; Remove Desktop shortcut
    Delete "$DESKTOP\${PRODUCT_NAME}.lnk"

    ; Remove registry keys
    DeleteRegKey HKLM "${PRODUCT_UNINST_KEY}"
    DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"

    ; Remove installed files
    RMDir /r "$INSTDIR"
SectionEnd

;---------------------------------------------------------------------------
; Functions
;---------------------------------------------------------------------------
Function .onInit
    ; Check if already installed
    ReadRegStr $0 HKLM "${PRODUCT_UNINST_KEY}" "UninstallString"
    ${If} $0 != ""
        MessageBox MB_OKCANCEL|MB_ICONINFORMATION \
            "${PRODUCT_NAME} is already installed. Click OK to uninstall the previous version first, or Cancel to abort." \
            IDOK uninst
            Abort
        uninst:
            ExecWait '$0 /S'
    ${EndIf}
FunctionEnd
