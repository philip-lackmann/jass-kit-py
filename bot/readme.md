### Deploy the bot                                                        
Connect from PowerShell:                                                  
1. `ssh -i path\to\key.pem ubuntu@<elastic-ip>`
2. `git pull origin`
3. `cd jass-kit-py/bot`      
4. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
5. `sudo cp deploy/jass-bot.service /etc/systemd/system/`
6. `sudo systemctl daemon-reload && sudo systemctl enable --now jass-bot`
                                                                        
Check that it works:
1. Reachability: run `curl http://<elastic-ip>:8888/rule_based` from    
your laptop. It should return Player rule_based here.   
2. Reboots: run `sudo reboot` on the server, wait a minute, and repeat the  
check.                                             