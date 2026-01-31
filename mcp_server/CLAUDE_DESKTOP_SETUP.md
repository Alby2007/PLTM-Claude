# Claude Desktop Configuration for PLTM MCP Server

## 📍 Step-by-Step Setup

### Step 1: Locate Claude Desktop Config File

**Windows Location:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Full path (typically):**
```
C:\Users\alber\AppData\Roaming\Claude\claude_desktop_config.json
```

### Step 2: Open the Config File

**Option A: Using File Explorer**
1. Press `Win + R`
2. Type: `%APPDATA%\Claude`
3. Press Enter
4. Open `claude_desktop_config.json` in a text editor

**Option B: Using Command Line**
```powershell
# Navigate to the directory
cd $env:APPDATA\Claude

# Open with notepad
notepad claude_desktop_config.json
```

**Option C: Direct path**
```powershell
notepad "C:\Users\alber\AppData\Roaming\Claude\claude_desktop_config.json"
```

### Step 3: Edit the Config File

**If the file is empty or has `{}`:**

Replace the entire contents with:

```json
{
  "mcpServers": {
    "pltm-memory": {
      "command": "python",
      "args": [
        "C:/Users/alber/CascadeProjects/LLTM/mcp_server/pltm_server.py"
      ]
    }
  }
}
```

**If the file already has other MCP servers:**

Add the PLTM server to the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "existing-server": {
      "command": "...",
      "args": ["..."]
    },
    "pltm-memory": {
      "command": "python",
      "args": [
        "C:/Users/alber/CascadeProjects/LLTM/mcp_server/pltm_server.py"
      ]
    }
  }
}
```

### Step 4: Save and Close

1. Save the file (Ctrl+S)
2. Close the text editor

### Step 5: Restart Claude Desktop

1. Completely close Claude Desktop (check system tray)
2. Reopen Claude Desktop
3. The PLTM tools should now be available

---

## ✅ Verification

After restarting Claude Desktop, you should see PLTM tools available.

**Test in Claude Desktop:**

```
You: "Can you use the PLTM tools?"

Claude should be able to call:
- store_memory_atom
- query_personality
- detect_mood
- get_mood_patterns
- resolve_conflict
- extract_personality_traits
- get_adaptive_prompt
- get_personality_summary
```

---

## 🐛 Troubleshooting

### Issue 1: Config file doesn't exist

**Solution:**
```powershell
# Create the directory if it doesn't exist
New-Item -ItemType Directory -Path "$env:APPDATA\Claude" -Force

# Create the config file
@"
{
  "mcpServers": {
    "pltm-memory": {
      "command": "python",
      "args": [
        "C:/Users/alber/CascadeProjects/LLTM/mcp_server/pltm_server.py"
      ]
    }
  }
}
"@ | Out-File -FilePath "$env:APPDATA\Claude\claude_desktop_config.json" -Encoding UTF8
```

### Issue 2: Tools not showing up

**Check:**
1. Config file is valid JSON (use https://jsonlint.com/)
2. Path to pltm_server.py is correct
3. Python is in your PATH
4. Claude Desktop was fully restarted

**Debug:**
```powershell
# Test if the server starts
python C:/Users/alber/CascadeProjects/LLTM/mcp_server/pltm_server.py
```

### Issue 3: Python not found

**Solution:**
Use full path to Python:

```json
{
  "mcpServers": {
    "pltm-memory": {
      "command": "C:/Users/alber/AppData/Local/Programs/Python/Python314/python.exe",
      "args": [
        "C:/Users/alber/CascadeProjects/LLTM/mcp_server/pltm_server.py"
      ]
    }
  }
}
```

---

## 📋 Quick Copy-Paste

**Complete config (ready to paste):**

```json
{
  "mcpServers": {
    "pltm-memory": {
      "command": "python",
      "args": [
        "C:/Users/alber/CascadeProjects/LLTM/mcp_server/pltm_server.py"
      ]
    }
  }
}
```

---

## 🎯 Next Steps

After configuration:

1. ✅ Config file created/updated
2. ✅ Claude Desktop restarted
3. ✅ Tools available in Claude Desktop
4. 📝 Start testing with the protocol
5. 🎉 Watch personality emerge!

---

**Status:** Ready for Claude Desktop integration ✅
