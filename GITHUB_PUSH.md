# Push to GitHub (one-time setup)

Your Mac has commits ready to push; GitHub just needs authentication.

If `git push` times out on port 22, use **HTTPS + token** (below) or SSH over port 443 (`HostName ssh.github.com`, `Port 443` in `~/.ssh/config`).

## Step 1 — Add SSH key to GitHub

1. Open **https://github.com/settings/keys**
2. Click **New SSH key**
3. Title: `Mac Wingspan` (any name)
4. Key type: **Authentication Key**
5. Paste this **entire line** (your public key):

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDVqz6ckgK6AUcLnPEA/9THcfrexG4xXYB15l1tY0W9c marlonbarrios-spread-your-wings
```

6. Click **Add SSH key**

## Step 2 — Push

In Terminal:

```bash
cd /Users/mbarriossolano/Desktop/spread_your_wings-main
git push origin main
```

Success looks like:

```text
Your branch is up to date with 'origin/main'.
```

Then refresh: **https://github.com/marlonbarrios/spread_your_wings**

## Already use HTTPS + token?

```bash
git remote set-url origin https://github.com/marlonbarrios/spread_your_wings.git
./push.sh YOUR_GITHUB_TOKEN
```

## Check status anytime

```bash
git status
```
