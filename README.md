# AirPods Auto-Translation with Python

Build your own AI speech translator using Python, Faster-Whisper, M2M100, and Piper TTS.

Speak into your computer and hear the translation. You do not press a record button. After a one-time setup, the AI models run on your computer. You do not need a paid API key.

The program runs on the computer. It does not install anything on AirPods. AirPods are optional. A normal microphone and speakers work too.

One translation model handles many languages, including English, Spanish, French, Vietnamese, Japanese, German, Korean, and Chinese. It handles one finished phrase at a time. It is not live interpreting, and it is not Apple's AirPods translation feature.

Speech recognition, translation, and spoken playback are three separate pieces. A language can be translated as text even when the computer cannot say that language out loud. The table below is the one to trust.

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

On a tested Windows PC, the Python libraries used about **1 GB**. The speech model is about **145 MB**. The translation model is about **1.9 GB**. Each spoken voice is about **63 MB**. Leave about **6 GB** free so the installer and the models have room.

The translation model stays in memory while the program is running. On the test PC it peaked at about **2.5 GB** of RAM with the speech model loaded too. A computer with **8 GB of RAM** has room for that plus Windows. **16 GB** is more comfortable if a browser is open too. The program does not need an NVIDIA graphics card.

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

1. Open [https://github.com/Nominally-tech/auto_translation](https://github.com/Nominally-tech/auto_translation).
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Find the ZIP in your Downloads folder. Right-click it and choose **Extract All**.
5. Open the extracted folder. GitHub usually names it something like `repository-name-main`. You should see `translator.py` and `README.md` inside.

### Option B — Clone with Git

Use this only if Git is already installed.

```powershell
git clone https://github.com/Nominally-tech/auto_translation.git
cd auto_translation
```

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

1. **Faster-Whisper** turns your speech into text. One model understands many languages. It writes down what you said. It does not translate.
2. **M2M100** translates that text. One model covers every language pair in this project. You do not download a new translation model when you change languages.
3. **Piper** speaks the translation out loud, when a voice exists for the language you want to hear.

Download the English-to-Spanish set once:

```powershell
.\.venv\Scripts\python.exe translator.py --download-models --source en --target es
```

You need internet for this step. That command downloads:

- Whisper `base`: about 145 MB. This same file is used for Japanese, French, Vietnamese, and the other spoken languages.
- M2M100 translation weights: about 1.9 GB, plus a small tokenizer. This file is used for every direction.
- The Spanish Piper voice: about 63 MB. Only the voice for `--target` is downloaded.

The files are saved in the `models` folder. You do not move them by hand. The next time you run the program, it reuses those files instead of downloading them again.

When it finishes, the last lines are:

```text
Models downloaded and loaded successfully.
The translation loop can now run locally, without sending audio to an API.
```

If the download stops halfway, run the same command again. Completed files are kept. A broken voice file is removed so it can be downloaded again.

Hearing English, French, Vietnamese, Japanese, German, or Korean requires that language's voice. Example for Japanese speech into English audio, which uses the English voice:

```powershell
.\.venv\Scripts\python.exe translator.py --download-models --source ja --target en
```

If the translation model is already on disk, this downloads the English voice and does not download the 1.9 GB file again. The same is true when you later pick French or another spoken target. Chinese has no voice in this project, so use `--no-tts` for Chinese output. The translation model still downloads.

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

You should also see `Speaking translation...` and hear the Spanish sentence. On the tested CPU run, this sentence became:

```text
Translated: Hola, ¿cómo estás tú?
```

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

On the test PC, with the models already downloaded, loading the translator took about **1 to 5 seconds**. A short sentence then translated in about **half a second** to **one second**. Recognizing a short recording took **under a second**. Turning text into speech took about **2 seconds** once that voice had been used, and closer to **20 seconds** the first time a voice was loaded. With the speech model and the translator loaded, the program peaked at about **2.5 GB** of RAM. Your computer can be slower or faster. This is not simultaneous interpreting.

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

`--source` is the language you speak. `--target` is the language you want back.

See every code:

```powershell
.\.venv\Scripts\python.exe translator.py --list-languages
```

English to Spanish, with speech:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es
```

Japanese, French, or Vietnamese into spoken English:

```powershell
.\.venv\Scripts\python.exe translator.py --source ja --target en
.\.venv\Scripts\python.exe translator.py --source fr --target en
.\.venv\Scripts\python.exe translator.py --source vi --target en
```

English into spoken French:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target fr
```

Spanish into English:

```powershell
.\.venv\Scripts\python.exe translator.py --source es --target en
```

Typed text, with no microphone and no speech. This is the easiest way to check Japanese, Korean, or Chinese:

```powershell
.\.venv\Scripts\python.exe translator.py --source ja --target en --text "こんにちは" --no-tts
```

Optional automatic language detection. You still choose the language you want to hear. If the program is not sure, it says so and does not guess a translation. Choosing `--source` yourself is more reliable for a demonstration.

```powershell
.\.venv\Scripts\python.exe translator.py --source auto --target en
```

Press **Ctrl+C** in the terminal to stop.

### Which directions speak, and which only print text

Checked on this project means the speech model accepts the language, the translation model accepts the pair, and a Piper voice exists for the language you hear.

| Direction | You speak | Translation | Computer speaks | What you can run |
| --- | --- | --- | --- | --- |
| Japanese → English | yes | yes | yes | full speech |
| French → English | yes | yes | yes | full speech |
| Vietnamese → English | yes | yes | yes | full speech |
| Spanish → English | yes | yes | yes | full speech |
| German → English | yes | yes | yes | full speech |
| Korean → English | yes | yes | yes | full speech |
| Chinese → English | yes | yes | yes | full speech |
| English → Spanish | yes | yes | yes | full speech |
| English → French | yes | yes | yes | full speech |
| English → Vietnamese | yes | yes | yes | full speech |
| English → German | yes | yes | yes | full speech |
| English → Korean | yes | yes | yes | full speech |
| English → Japanese | yes | yes | text only | add `--no-tts` |
| English → Chinese | yes | yes | text only | add `--no-tts` |

Japanese and Chinese text can be translated. This project does not speak those two languages. Piper's Japanese voice needs the `pyopenjtalk` package, and that package has no Windows installer for Python 3.12. The published Piper card for Chinese does not state a license. In both cases the words still print with `--no-tts`. Hearing English, Spanish, French, Vietnamese, German, or Korean still works.

M2M100 can also translate many other language codes, such as `pt` or `hi`. Those codes work as text with `--no-tts`. Do not expect spoken output unless `--list-languages` says that target has a spoken reply. The program will not read a foreign sentence with an English voice.

The first time you pick a new spoken target, that voice downloads. The 1.9 GB translation model does not download again.

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

**Fix:** The translation model is large, and the pause after a sentence is normal on CPU. Keep `--asr-model base` unless you need to save time. A smaller speech model starts the pause sooner:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es --asr-model tiny
```

`tiny` is faster and makes more mistakes. `base` is the default. `small`, `medium`, and `large-v3` are slower in that order. There is no GPU step in the normal install. Leave out `--device cuda` unless you installed a CUDA build of PyTorch on purpose.

### 17. The target language cannot be spoken

**Problem:** The program says no verified Piper voice is available, or you asked for Chinese speech.

**Why it happens:** Translation and speech are separate. Chinese text can be translated. This project has no Chinese voice.

**Fix:** Print the translation without audio:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target zh --text "Hello, how are you?" --no-tts
```

Use `--list-languages` to see which targets can be spoken.

### 18. A plain PyTorch install replaced the CPU build

**Problem:** An import mentions CUDA, or the version does not end in `+cpu`.

**Why it happens:** `pip install torch` without the CPU link downloads a different build.

**Fix:** Repeat the three commands in Step 4. Keep the `--index-url` line.

### 19. PowerShell blocks virtual environment activation

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

- `asr.py` listens for a phrase and turns it into text in the language you chose.
- `nmt.py` translates that text with one multilingual model.
- `tts.py` turns the translation into spoken audio when a voice exists for that language.
- `translator.py` is the program you run. It connects the other three.

## Customization (optional)

You can ignore this section until the basic translator works.

Basic choices:

```powershell
.\.venv\Scripts\python.exe translator.py --source en --target es
.\.venv\Scripts\python.exe translator.py --source es --target en
.\.venv\Scripts\python.exe translator.py --source ja --target en
.\.venv\Scripts\python.exe translator.py --source fr --target en
.\.venv\Scripts\python.exe translator.py --source vi --target en
.\.venv\Scripts\python.exe translator.py --source en --target fr
.\.venv\Scripts\python.exe translator.py --source auto --target en
.\.venv\Scripts\python.exe translator.py --list-languages
.\.venv\Scripts\python.exe translator.py --list-devices
.\.venv\Scripts\python.exe translator.py --source en --target es --input-device 1 --output-device 3
.\.venv\Scripts\python.exe translator.py --source en --target es --text "Hello, how are you?"
.\.venv\Scripts\python.exe translator.py --source ja --target en --text "こんにちは" --no-tts
.\.venv\Scripts\python.exe translator.py --source en --target es --asr-model small
.\.venv\Scripts\python.exe translator.py --help
```

`--asr-model` can be `tiny`, `base`, `small`, `medium`, or `large-v3`. `base` is the normal choice. The normal install is CPU-only. Do not pass `--device cuda` unless you have installed a CUDA build of PyTorch yourself.

Helpful extras, if the basic run is already working:

- `--no-tts` prints the translation and does not speak. Use it for Chinese output and for any M2M100 language that has no voice.
- `--speech-threshold` sets how loud speech must be. Leave it out to measure the room automatically.
- `--silence-duration` is how long a pause must last before the phrase ends. The default is `0.8` seconds.
- `--playback-gap` is the quiet wait after speech before listening again. The default is `0.4` seconds.
- `--source auto` asks Whisper which language it heard. A short or unclear phrase is skipped instead of translated from the wrong language.

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
Yes. Use `--source es --target en`. The translation model is the same file you already downloaded. The first run in that direction also downloads the English voice if it is not there yet.

**Can I translate Japanese, French, or Vietnamese?**
Yes. Japanese, French, and Vietnamese into spoken English work. English into spoken French and Vietnamese works too. English into Japanese or Chinese prints the words and does not speak them. Add `--no-tts` for those.

**Can I add other languages?**
Text translation already accepts the other M2M100 codes. Run `--list-languages`. Spoken output exists only for the voices listed there. Another spoken language needs a real Piper voice checked in `tts.py`. Do not point an English voice at a foreign sentence.

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
| Translation model | [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) | MIT. Paper: Fan et al., 2020, [arXiv:2010.11125](https://arxiv.org/abs/2010.11125) |
| Speech synthesis | [Piper](https://github.com/OHF-Voice/piper1-gpl) (`piper-tts`) | GPL-3.0-or-later |
| English voice | [`en_US-ljspeech-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ljspeech/medium/MODEL_CARD) | Public-domain [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) data, trained from scratch |
| Spanish voice | [`es_ES-davefx-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/es/es_ES/davefx/medium/MODEL_CARD) | The model card lists the dataset as CC0 and says the voice was finetuned from Lessac |
| French voice | [`fr_FR-siwis-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/fr/fr_FR/siwis/medium/MODEL_CARD) | SIWIS dataset is CC-BY 4.0. The card says the voice was finetuned from Lessac |
| Vietnamese voice | [`vi_VN-vais1000-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/vi/vi_VN/vais1000/medium/MODEL_CARD) | VAIS-1000 is CC-BY 4.0. The card says the voice was finetuned from Lessac |
| German voice | [`de_DE-thorsten-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/de/de_DE/thorsten/medium/MODEL_CARD) | Thorsten dataset is CC0. The card says the voice was finetuned from Lessac |
| Korean voice | [`ko_KR-kss-medium`](https://huggingface.co/rhasspy/piper-voices/blob/main/ko/ko_KR/kss/medium/MODEL_CARD) | Korean Single Speaker dataset is CC BY-NC-SA 4.0 (non-commercial) |

Copyright (C) 2026 Nominally.

AirPods is a name used here only to describe headphones you can connect to the computer. This project is not affiliated with Apple, Hugging Face, SYSTRAN, or the Open Home Foundation.
