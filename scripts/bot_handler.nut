// https://developer.valvesoftware.com/wiki/Team_Fortress_2/Scripting/Script_Functions/Constants
::TF_TEAM_RED <- 2
::NO_MISSION <- 0

class::bot_handler {
	bot_list = []

    // mp_autoteambalance 0
    // mp_teams_unbalance_limit 0
    // nb_player_move 0
    // tf_bot_add 1 demoman red
    // tf_bot_keep_class_after_death 1
    // tf_bot_fire_weapon_allowed 0
    // tf_bot_force_class demoman
    // tf_bot_quota_mode match

	// w    e cannot add bot via SpawnEntityFromTable() bcs it crashes ;3
	function AddTFBot() {
		// This triggers the point_servercommand to run tf_bot_add
        //SendToServerConsole("tf_bot_add 1 demoman red")
	}

	constructor(bots_num) {
		//adding all bots
		for (local i = 0; i < bots_num; i++) {
			AddTFBot()
		}

        bot_list = []
        // getting reference to them:
		local ent = null
		while (ent = Entities.FindByClassname(ent, "player")) { //our bots are player class
            // lets assume that bots are in RED team
            if (ent.GetTeam() == TF_TEAM_RED) {
                bot_list.append(ent)
            }
		}
    }

    function GetBots() {

		local ent = null
		while (ent = Entities.FindByClassname(ent, "player")) { //our bots are player class
            //printl(ent)
            // lets assume that bots are in RED team
            if (ent.GetTeam() == TF_TEAM_RED) {
                bot_list.append(ent)
            }
		}
        return bot_list
    }

    function PrintBots() {
        if (bot_list ==  null) {
            return
        }
        foreach(i, ent in bot_list) {
            printl(i)
            printl(ent)
        }
    }

    function MoveBots() {
        foreach(i, ent in bot_list) {
            ent.SetVelocity(Vector(200.0,200.0,200.0))
        }
        printl("moved players")
    }

    function MakeBotsFire() {
        bot_list = GetBots()
        foreach(i, ent in bot_list) {
            ent.GetActiveWeapon().PrimaryAttack()
        }
        printl("Bots attacked")
    }

    function RotateBots() {
        foreach(i, ent in bot_list) {
            local eye = ent.LocalEyeAngles()
            //eye.x += 10 // obrot w pionie
            printl(eye)
            eye.y += 20 // obrot w poziomie
            ent.SnapEyeAngles(eye)
        }
        printl("bot rotaed")
    }

    function TeleportBots(position) {
        foreach(i, ent in bot_list) {
            local pos = ent.GetLocalOrigin()
            pos.x += i * 30
            ent.SetLocalOrigin(pos)
        }
    }


    function UpdateScript() {
        bot_list = GetBots()
        foreach(key,bot in bot_list) {
            bot.ValidateScriptScope()
            local scope = bot.GetScriptScope()
            printl(scope)
            scope.brain <- this
            scope.Think <- function() {
                return brain.Update()
            }
            AddThinkToEnt(bot,"Think")
        }
    }

    function ShowScripts() {
        bot_list = GetBots()
        foreach(key,bot in bot_list) {
            local scope = bot.GetScriptScope()
            printl(scope)
        }

    }

    function brain() {
    }

    function Update() {
        //ent.SetAbsAngles(QAngle(eye.x, eye.y, eye.z))
        printl("thinking")
    }

    function RotateBot(bot_id, yaw, pitch) {
        bot_list = GetBots()
        if(bot_id < 0 || bot_id >= bot_list.len()-1) {
            printl("Bad index")
            return;
        }

        local ent = bot_list[bot_id]

        if (!ent) {
            printl("RotateBot: invalid entity.");
            return;
        }

        local eye = ent.LocalEyeAngles();

        eye.x = pitch; // Pitch (up/down)
        eye.y = yaw;   // Yaw (left/right)

        ent.SnapEyeAngles(QAngle(eye.x, eye.y, 0));
    }

    function BotIgnoreEnemy() {

        foreach(key,bot in bot_list) {
            if (!bot.HasBotAttribute(1024)) {
                bot.AddBotAttribute(1024)
            }
        }
    }
}


bot_handler <- ::bot_handler(4)
bot_handler.PrintBots()
bot_handler.BotIgnoreEnemy()
//bot_handler.MoveBots()
//bot_handler.MakeBotsFire()
bot_handler.RotateBots()
bot_handler.TeleportBots(Vector(500,0,200))
//bot_handler.UpdateScript()
//bot_handler.ShowScripts()

//function AddBot()
//{
//    // This triggers the point_servercommand to run tf_bot_add
//    EntFire("servercmd", "Command", "tf_bot_add 1 sniper blue");
//}

//AddBot()

// function ExecuteServerCommand()
// {
//     printl("[VScript] Script started");
//     local command = "tf_bot_add 1 demoman red";
//     EntFire("point_servercommand", "Command", command, 0.0, null);
//     printl("[VScript] Command sent: " + command);
// }

// ExecuteServerCommand();