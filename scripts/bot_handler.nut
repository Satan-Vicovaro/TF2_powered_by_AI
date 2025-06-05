// https://developer.valvesoftware.com/wiki/Team_Fortress_2/Scripting/Script_Functions/Constants
::TF_TEAM_RED <- 2
::TF_TEAM_BLUE <- 3
::MIN_RADIUS <- 1
::MAX_RADIUS <- 2
::BOT_TYPE <- "soldier"

IncludeScript("projectile_handler")

class::bot_handler {
	bot_list = []

	constructor() {
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
        foreach(i, ent in bot_list) {
            local weapon = ent.GetActiveWeapon()
            weapon.PrimaryAttack()
            weapon.SetClip1(weapon.GetMaxClip1())
        }
        printl("Bots attacked")
    }

    // Places bots evenly around a circle of a given center point and radius
    //
    function TeleportBots(center_pos, min_r, max_r) {
        foreach(i, ent in bot_list) {
            local radius = rand() % (max_r - min_r) + min_r
            local angle = 2 * PI * i / bot_list.len()
            local bot_pos = ent.GetLocalOrigin()

            bot_pos.x = center_pos.x + radius * cos(angle)
            bot_pos.y = center_pos.y + radius * sin(angle)
            ent.SetLocalOrigin(bot_pos)
        }
    }

    function RotateBot(bot_id, yaw, pitch) {
        local ent = null

        foreach(i, bot in bot_list) {
            if (bot.entindex() == bot_id) {
                ent = bot
                break
            }
        }

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

    function Setup() {
        BotIgnoreEnemy()
        TeleportBots(Vector(0,0,0), MIN_RADIUS, MAX_RADIUS)
    }
}

bot_handler <- ::bot_handler()

function ClearStringFromPool(string)
{
	local dummy = Entities.CreateByClassname("info_target")
	dummy.KeyValueFromString("targetname", string)
	NetProps.SetPropBool(dummy, "m_bForcePurgeFixedupStrings", true)
	dummy.Destroy()
}

function EntFireCodeSafe(entity, code, delay = 0.0, activator = null, caller = null)
{
	EntFireByHandle(entity, "RunScriptCode", code, delay, activator, caller)
	ClearStringFromPool(code)
}


local EventsID = UniqueString()
getroottable()[EventsID] <-
{
    // Example usage:
    //
    // FireScriptHook("Set_Angles", {
    //     data = [
    //         {id=2, x=50.0, y=50.0},
    //         {id=3, x=67.0, y=76.0},
    //     ]
    // })
    OnScriptHook_Set_Angles = function(params) {
        foreach (i, data in params.data) {
            bot_handler.RotateBot(data.id, data.y, data.x)
        }

        EntFireCodeSafe(GetListenServerHost(), "bot_handler.MakeBotsFire()", 0.3)
        //bot_handler.MakeBotsFire()
    }

    OnScriptHook_Change_Pos = function(_) {
        bot_handler.TeleportBots(Vector(0,0,140), MIN_RADIUS, MAX_RADIUS)
    }

	// Cleanup events on round restart
	OnScriptHook_Kill_BotHandler = function(_) {
        printl("Deleting hooks for bot_handler")
        delete getroottable()[EventsID]
    }

    OnScriptHook_SendProjectileInfo = function(_) {
       send_projectile_info()
    }
}
local EventsTable = getroottable()[EventsID]
foreach (name, callback in EventsTable) EventsTable[name] = callback.bindenv(this)
__CollectGameEventCallbacks(EventsTable)