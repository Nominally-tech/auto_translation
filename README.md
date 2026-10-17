# AirPods Auto-Translation with Python

Build your own AI speech translator using Python, Faster-Whisper, MarianMT, and Piper TTS.

Speak English into your computer and hear the Spanish translation. You do not press a record button. After a one-time setup, the AI models run on your computer. You do not need a paid API key.

The program runs on the computer. It does not install anything on AirPods. AirPods are optional. A normal microphone and speakers work too.

It translates English to Spanish, and Spanish to English. It handles one finished phrase at a time. It is not live interpreting, and it is not Apple's AirPods translation feature.

```text
Microphone → Speech Recognition → Translation → Speech Generation → Headphones
```

## What you'll need

- A Windows computer. That is the setup this guide is written for.
- Python 3.12. Not Python 3.14.
- An internet connection while you install the libraries and download the models. After that, translation can stay on your computer.
- A microphone that Windows can already hear.
- Speakers or headphones.
- AirPods or other Bluetooth earbuds, only if you want them.

You do not need an NVIDIA graphics card. The normal install uses the computer's processor (the CPU). A faster computer will answer sooner. A slower computer will still work, with a longer pause after you stop speaking.

On a tested Windows PC, the Python libraries used about **1 GB**. The English-to-Spanish models use about **530 MB** more. Leave about **4 GB** free so the installer has room.

## Step 1: Download Python

Python is the program that runs this project.

1. Open [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
2. Find **Python 3.12**. Do not click a Python 3.13 or 3.14 download. The big button on the Python homepage is often a newer version than this project uses.
3. Download the **Windows installer (64-bit)**.
4. Run the installer.
5. On the first screen, turn on **Add python.exe to PATH** if you see that checkbox. Then click **Install Now**.

PATH tells Windows where Python lives. Turning it on now saves a later problem.

6. Open PowerShell. You can also use the terminal inside VS Code after you open the project in Step 2.
7. Check that Python 3.12 is available:

```powershell
py -3.12 --version
```

You should see a version that starts with 3.12, such as:

```text
Python 3.12.x
```

The last part can be any number. It must start with `Python 3.12`.

**If that command fails,** close PowerShell, open it again, and retry. If it still fails, Python 3.12 is not installed, or Windows is opening the Microsoft Store instead. Go to **Settings → Apps → Advanced app settings → App execution aliases** and turn off the aliases named `python.exe` and `python3.exe`. Then install Python 3.12 again and retry `py -3.12 --version`.

## Step 2: Download the project

Put the project folder somewhere easy to find, such as Documents.

### Option A — Download ZIP (easiest)

1. Open this project's page on GitHub.
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Find the ZIP in your Downloads folder. Right-click it and choose **Extract All**.
5. Open the extracted folder. GitHub usually names it something like `repository-name-main`. You should see `translator.py` and `README.md` inside.

### Option B — Clone with Git

Use this only if Git is already installed.

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd auto_translation
```

Replace the URL with the real GitHub address when the repository exists. `YOUR_USERNAME` and `YOUR_REPOSITORY` are placeholders.

`cd auto_translation` matches this project's folder name. If you named the GitHub repository something else, type `cd` and then that folder's name instead.

### Open the folder and a terminal

1. Open VS Code.
2. Choose **File → Open Folder** and select the folder that contains `translator.py`.
3. Choose **Terminal → New Terminal**.

The terminal must be inside that folder. If you type `dir`, you should see `translator.py` in the list. If you do not, you are in the wrong folder. Use `cd` to move into the project folder before the next step.

## Step 3: Create a Python environment

A virtual environment is a private folder for this project's libraries. It keeps them separate from other Python projects on your computer.

In the project folder, run:

```powershell
py -3.12 -m venv .venv
```

This creates a new folder named `.venv`. You do not need to open it or edit anything inside it. If you do not see `.venv` in VS Code, hidden folders may be hidden. That is fine. The folder is still there.

You do **not** have to activate the environment. Every later command starts with:

```powershell
.\.venv\Scripts\python.exe
```

That line means "use the Python inside this project," so Windows does not grab a different Python.

### Activation is optional

If a tutorial tells you to activate the environment, the PowerShell command is:

```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt then starts with `(.venv)`. You can skip this.

**If PowerShell says running scripts is disabled,** do not change Windows security settings for the whole computer. Either keep using `.\.venv\Scripts\python.exe`, or allow scripts for this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

That permission lasts until you close the window.

## Step 4: Install the required libraries

Stay in the project folder. Run these three commands one at a time. Wait until each one finishes.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "pip>=23.1,<26"
```

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This can take several minutes. The libraries are large. A successful command ends with `Successfully installed ...`.

The first command keeps pip on a version that works with this project. A newer pip release crashed during testing. The second command installs the CPU version of PyTorch, which is what the translator uses. A normal `pip install torch` can download a much larger GPU package that you do not need.

Check that the important libraries can be imported:

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, faster_whisper, piper, sounddevice, numpy; print('Libraries installed'); print('PyTorch', torch.__version__)"
```

A successful check looks like this:

```text
Libraries installed
PyTorch 2.14.1+cpu
```

The PyTorch numbers may be newer. The version should end in `+cpu`. If you see an error that names a missing library, run the three install commands again.

## Step 5: Download the AI models

The translator uses three kinds of AI:

1. **Faster-Whisper** turns your speech into text.
2. **MarianMT** translates that text.
3. **Piper** speaks the translation out loud.

Download them once:

```powershell
.\.venv\Scripts\python.exe translator.py --download-models --source en --target es
```

You need internet for this step. The download is about 530 MB for English to Spanish:

- Whisper `base`: about 145 MB
- English → Spanish translation model: 312 MB, plus a small tokenizer
- Spanish Piper voice: about 63 MB

The files are saved in the `models` folder. You do not move them by hand. The next time you run the program, it reuses those files instead of downloading them again.

When it finishes, the last lines are:

```text
Models downloaded and loaded successfully.
The translation loop can now run locally, without sending audio to an API.
```

If the download stops halfway, run the same command again. Completed files are kept. A broken voice file is removed so it can be downloaded again.

Spanish to English needs one more translation model and the English voice, about 380 MB extra:

```powershell
.\.venv\Scripts\python.exe translator.py --download-models --source es --target en
```

You can skip that until you want Spanish-to-English. If you start that direction later without downloading first, the program downloads the missing files then.

## Step 6: Test the translator without a microphone

This checks the translation and the speakers before you deal with the microphone.

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --text "Hello, how are you?"
```

The first run loads the models. That can take a minute. Then you should see:

```text
Original: Hello, how are you?
Translated: Hola, ¿cómo estás?
```

You should also see `Speaking translation...` and hear the Spanish sentence. On a tested run, the model produced exactly `Hola, ¿cómo estás?`. A small difference in wording still means the translation worked.

Two different things can succeed:

- **Translation worked** if the original and translated lines appear.
- **Audio worked** if you also hear the Spanish sentence.

If you see the words but hear nothing, the models are fine and the speaker choice or volume is not. Try `--no-tts` to confirm the words without sound:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --text "Hello, how are you?" --no-tts
```

Then check the speaker volume and Step 8 if you are using headphones.

## Step 7: Start automatic translation

This uses your normal microphone and speakers. You do not pick device numbers for this first try.

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es
```

Then:

1. Leave the program running.
2. Stay quiet for a moment. It measures room noise and prints a line that starts with `Calibrated speech threshold`.
3. Wait until you see `Listening...`.
4. Say a short English sentence in a normal voice.
5. Stop talking and pause for about a second.
6. The terminal prints what you said.
7. It prints the Spanish translation.
8. The computer speaks the Spanish.
9. `Listening...` appears again by itself.

You do not press a record button. You do not restart the program between sentences.

Press **Ctrl+C** in the terminal to stop. You should see `Stopped.`

A successful phrase looks like this:

```text
===================================
 NOMINALLY - AUTO TRANSLATION
===================================

Source Language: English
Target Language: Spanish
Input Device: Default Microphone
Output Device: Default Speakers

Models loaded successfully.

Listening...

Original: Hello, how are you?
Translated: Hola, ¿cómo estás?

Speaking translation...

Listening...
```

There is a pause after you stop speaking. The computer has to recognize the words, translate them, and generate speech. That is normal.

## Step 8: Using AirPods or Bluetooth headphones

Skip this step if the computer's microphone and speakers are already fine.

AirPods are just another Windows audio device. This project does not install an app on them.

1. Put the AirPods in pairing mode.
2. Open **Settings → Bluetooth & devices → Add device** and connect them.
3. In the project folder, list the devices Python can use:

```powershell
.\.venv\Scripts\python.exe translator.py --list-devices
```

Each row has an index, an input count, an output count, and a name.

- A microphone has an input count greater than 0.
- Speakers or headphones have an output count greater than 0.

4. Start the translator with the indexes from **your** list. The numbers below are only an example. Yours will be different.

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --input-device 1 --output-device 3
```

In that example, `1` is the microphone and `3` is the headphones. Look at your own table before you type the numbers.

Two ways to use AirPods:

- **Clearer:** use the computer microphone, or a USB microphone, as the input. Use the AirPods as the output. On Windows the stereo AirPods device is often named `Headphones`.
- **All on the AirPods:** pick the AirPods row whose name contains `Hands-Free` for the microphone, and the matching hands-free row for the speakers.

Hands-free mode can make the sound thin and muffled. Windows switches many earbuds into that mode when an app opens the earbud microphone. For a clearer result, use the computer microphone plus the AirPods headphones.

The banner prints the device names you chose, so you can see whether the right ones were used.

## Changing translation languages

`--source` is the language you speak. `--target` is the language you hear.

English to Spanish:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es
```

Spanish to English:

```powershell
.\.venv\Scripts\python.exe translator.py --source es --target en
```

Only those two directions work. French, German, and other languages are not included.

## Troubleshooting

### 1. 'python' is not recognized

**Problem:** PowerShell says `python` is not recognized, or the Microsoft Store opens.

**Why it happens:** Windows does not know which Python you mean, or the Store alias is in the way.

**Fix:** Use `py -3.12` instead of `python`. If the Store opens, turn off the `python.exe` and `python3.exe` app execution aliases, then open a new PowerShell window.

### 2. 'py' is not recognized

**Problem:** `py -3.12 --version` says `py` is not recognized.

**Why it happens:** The Python launcher was not installed.

**Fix:** Run the Python 3.12 installer again. Choose **Modify**, and include the **py launcher**. Then close PowerShell and open it again.

### 3. Python 3.14 was selected instead of Python 3.12

**Problem:** `python --version` shows 3.14, or the program prints a warning that you are not on Python 3.12.

**Why it happens:** The newest Python from the homepage is not the version this project was tested with. Packages may have no install file for 3.14.

**Fix:** Install Python 3.12 from the Windows downloads page. Create the environment again:

```powershell
py -3.12 -m venv .venv
```

If `.venv` already exists, delete that folder first, then run the command again. After that, use `.\.venv\Scripts\python.exe` for every install and run command.

### 4. 'faster-whisper' is not installed

**Problem:** The program says `faster-whisper is not installed` or `A required library is not installed`.

**Why it happens:** Step 4 did not finish, or the command used a different Python than the `.venv` one.

**Fix:** From the project folder, run the three install commands in Step 4 again. Always start them with `.\.venv\Scripts\python.exe`.

### 5. The virtual environment does not exist

**Problem:** PowerShell says it cannot find `.\.venv\Scripts\python.exe`.

**Why it happens:** Step 3 has not been run, or the terminal is not inside the project folder.

**Fix:** Run `dir` and confirm you can see `translator.py`. Then run:

```powershell
py -3.12 -m venv .venv
```

### 6. PyTorch installation fails

**Problem:** The `torch` command stops with a network error, or a later import says the version is invalid.

**Why it happens:** The download was interrupted, or pip installed a GPU build or a packaging version that cannot read the CPU release.

**Fix:** Run these again, in this order:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "pip>=23.1,<26"
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The CPU index URL matters. Do not leave it out.

### 7. The wrong microphone is selected

**Problem:** The program listens, but it never hears you, or it hears the wrong room.

**Why it happens:** Windows has more than one microphone. The default may be a headset you are not wearing.

**Fix:**

```powershell
.\.venv\Scripts\python.exe translator.py --list-devices
```

Choose a row with an input count above 0 and a name you recognize. Start again with `--input-device` and that number.

### 8. The program shows Listening but never processes speech

**Problem:** `Listening...` stays on screen after you talk.

**Why it happens:** The microphone is too quiet, the wrong device is selected, or you did not pause long enough for the program to decide the phrase ended.

**Fix:** Pause for a full second after the sentence. Speak a little louder. If it still waits, lower the detection level:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --speech-threshold 0.01
```

Also check **Settings → Privacy & security → Microphone** and allow desktop apps to use the microphone.

### 9. The program repeatedly detects background noise

**Problem:** It translates when nobody is speaking, or it reacts to fans and keyboard noise.

**Why it happens:** The automatic noise measurement set the level too low for a noisy room.

**Fix:** Raise the level so only clearer speech counts:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --speech-threshold 0.03
```

Try `0.04` or `0.05` if it still triggers. You can also ignore very short sounds with `--min-speech-duration 0.6`.

### 10. The app translates its own spoken translation

**Problem:** After it speaks Spanish, it treats that audio as a new sentence.

**Why it happens:** The speakers are loud enough for the microphone to hear them. The program already closes the microphone while it is speaking, but sound can still leak in afterward.

**Fix:** Use headphones. Then wait longer before listening again:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --playback-gap 0.8 --silence-duration 1.2
```

### 11. The translation appears but no sound plays

**Problem:** You see `Translated:` and `Speaking translation...`, but you hear nothing.

**Why it happens:** The words were translated. Playback went to a device that is muted, unplugged, or not the one you are wearing.

**Fix:** Raise the Windows volume. Then list devices and pass the speaker row you actually want:

```powershell
.\.venv\Scripts\python.exe translator.py --list-devices
.\.venv\Scripts\python.exe translator.py --source en --target es --text "Hello, how are you?" --output-device 3
```

Change `3` to the output index from your list.

### 12. AirPods do not appear in the device list

**Problem:** `--list-devices` has no AirPods row.

**Why it happens:** They are not connected for audio, or Windows connected them only as a phone device.

**Fix:** In **Settings → Bluetooth & devices**, confirm the AirPods show Connected. Play any Windows sound and choose the AirPods in the volume panel. Then run `--list-devices` again. If they disconnected, put them in your ears and reconnect.

### 13. AirPods sound muffled

**Problem:** Speech through the AirPods sounds like a phone call.

**Why it happens:** Windows is using hands-free mode so the AirPods microphone can stay on. The device name usually contains `Hands-Free`.

**Fix:** Stop the program. Start it again with the computer microphone as `--input-device` and the `Headphones` AirPods row as `--output-device`.

### 14. Model downloads fail or stop partway through

**Problem:** The download command ends with a network error, or a later run says a model could not be loaded.

**Why it happens:** The connection dropped, or a firewall blocked `huggingface.co`. Piper voices also come from Hugging Face.

**Fix:** Connect to the internet and run the same `--download-models` command again. If one folder is clearly broken, delete only that folder and download again:

- `models\asr` for speech recognition
- `models\nmt` for translation
- `models\tts` for the voice

Do not delete the whole project.

### 15. The application is slow on CPU

**Problem:** You wait a long time after you stop speaking.

**Why it happens:** The default `base` speech model is a balance of speed and accuracy. Larger models are slower. CPU speed also varies by computer.

**Fix:** Use the smaller speech model:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --asr-model tiny
```

`tiny` is faster and makes more mistakes. `base` is the default. `small`, `medium`, and `large-v3` are slower in that order.

### 16. PowerShell blocks virtual environment activation

**Problem:** `Activate.ps1` fails with an execution-policy message.

**Why it happens:** PowerShell does not run scripts until you allow them.

**Fix:** You can ignore activation. Run commands with `.\.venv\Scripts\python.exe` instead. If you still want activation, use the one-window command in Step 3. Do not set the execution policy for the whole computer.

## Project structure

```text
auto_translation/
├── asr.py
├── nmt.py
├── tts.py
├── translator.py
├── requirements.txt
├── README.md
├── LICENSE
├── models/          # downloaded AI files, not uploaded to GitHub
├── outputs/         # optional recordings, not uploaded to GitHub
└── tests/
```

- `asr.py` listens for a phrase and turns it into text.
- `nmt.py` translates that text between English and Spanish.
- `tts.py` turns the translation into spoken audio.
- `translator.py` is the program you run. It connects the other three.

## Customization (optional)

You can ignore this section until the basic translator works.

Basic choices:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es
.\.venv\Scripts\python.exe translator.py --source es --target en
.\.venv\Scripts\python.exe translator.py --list-devices
.\.venv\Scripts\python.exe translator.py --source en --target es --input-device 1 --output-device 3
.\.venv\Scripts\python.exe translator.py --source en --target es --text "Hello, how are you?"
.\.venv\Scripts\python.exe translator.py --source en --target es --asr-model small
.\.venv\Scripts\python.exe translator.py --help
```

`--asr-model` can be `tiny`, `base`, `small`, `medium`, or `large-v3`. `base` is the normal choice.

Helpful extras, if the basic run is already working:

- `--no-tts` prints the translation and does not speak.
- `--speech-threshold` sets how loud speech must be. Leave it out to measure the room automatically.
- `--silence-duration` is how long a pause must last before the phrase ends. The default is `0.8` seconds.
- `--playback-gap` is the quiet wait after speech before listening again. The default is `0.4` seconds.

Adding another language means editing `nmt.py` and `tts.py` after you have tested a real translation model and a real Piper voice. Do not add a language name until you have heard it work. The program rejects directions that are not English ↔ Spanish.

## FAQ

**Do I need AirPods?**
No. Any microphone and speakers that Windows can use will work.

**Is this free?**
Yes. There is no API key and no subscription. You still need internet once, to download the free libraries and models.

**Does it work offline?**
After Step 4 and Step 5, translation runs on your computer. You do not need internet for each sentence.

**Do I need a GPU?**
No. The install in this guide uses the CPU.

**Does it work with regular headphones?**
Yes. Plug them in, or connect them, and use the computer's default speakers. If the sound goes to the wrong place, pick the headphones with `--output-device`.

**Does it automatically listen?**
Yes. After `Listening...` appears, speak and then pause. It listens again after it finishes speaking.

**How do I stop it?**
Click the terminal and press Ctrl+C.

**Can I translate Spanish into English?**
Yes. Use `--source es --target en`. Download those models first if you have not already.

**Can I add other languages?**
Not from the command line. Only English and Spanish are set up. Another language needs a tested translation model and a tested Piper voice added in the code.

**Can I use it on my phone?**
No. This is a computer program. The phone is not where the models run.

**Is this the official Apple AirPods translation feature?**
No. This is an independent Python project for the Nominally tutorial. Apple does not make it, and it does not run inside the AirPods.

## Credits and licenses

The tutorial code in this repository is **GPL-3.0-or-later**. See [LICENSE](LICENSE). The code calls Piper, and Piper itself is GPL-3.0-or-later, so a copy of this program that you give to other people stays under the GPL.

The downloaded models and voices are **not** part of that license. They stay under the licenses on their own pages, and they are not stored in git.

| Piece | Project | License |
| --- | --- | --- |
| Speech recognition | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), running OpenAI Whisper models | MIT code. Model cards: [tiny](https://huggingface.co/Systran/faster-whisper-tiny), [base](https://huggingface.co/Systran/faster-whisper-base), [small](https://huggingface.co/Systran/faster-whisper-small), [medium](https://huggingface.co/Systran/faster-whisper-medium), [large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) |
| Translation code | [Hugging Face Transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| Translation models | [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) and [Helsinki-NLP/opus-mt-es-en](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) | Apache-2.0 on both model cards |
| Speech synthesis | [Piper](https://github.com/OHF-Voice/piper1-gpl) (`piper-tts`) | GPL-3.0-or-later |
| English voice | [`en_US-ljspeech-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ljspeech/medium/MODEL_CARD) | Public-domain [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) data, trained from scratch |
| Spanish voice | [`es_ES-davefx-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/es/es_ES/davefx/medium/MODEL_CARD) | The model card lists the dataset as CC0 and says the voice was finetuned from Lessac. Read the card before sharing the voice file |

Copyright (C) 2026 Nominally.

AirPods is a name used here only to describe headphones you can connect to the computer. This project is not affiliated with Apple, Hugging Face, SYSTRAN, or the Open Home Foundation.
