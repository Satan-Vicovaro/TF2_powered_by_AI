//Globals
::FLT_MAX <- 3.402823466e+38
::TF_TEAM_RED <- 2
::TF_TEAM_BLUE <- 3
::MOVETYPE_NOCLIP <- 8
::MOVECOLLIDE_DEFAULT <- 0
::IGNORE_ENEMIES <- 1024
::STATUE_HEIGHT <- 0

// Debugging constant
const debug = false

// Bot spawn constants
const DMG_BEFORE_REPOSITION = 100 // TODO specify how much damage for bot to reposition
const SPAWN_RADIUS = 200
const ORIGIN_X = 0
const ORIGIN_Y = 0
const MAX_HEALTH = 10000
const GRAV_MIN = 0.00001

IncludeScript("vs_math")
IncludeScript("file_scripts")

class ::TargetBot
{
    bot = null
    health = MAX_HEALTH
    reposition_damage = DMG_BEFORE_REPOSITION
    auto_reposition = false
    damage_register = ""

    constructor()
    {
        // get a blue bot
        local bot_list = []
        local ent = null
        while (ent = Entities.FindByClassname(ent, "player")) {
            if (ent == GetListenServerHost()) continue

            if (ent.GetTeam() == TF_TEAM_BLUE) {
                bot_list.append(ent)
            }
        }
        this.bot = bot_list[0]

        // setting attributes
        this.bot.SetGravity(GRAV_MIN)
        this.bot.SetHealth(health)
        this.bot.SetLocalOrigin(Vector(0, 0, 0))
        this.bot.SetMoveType(MOVETYPE_NOCLIP,  MOVECOLLIDE_DEFAULT)
        this.bot.AddBotAttribute(1024)

        // initial state
        this.move_to_random_position()
    }

    // moves bot to destination Vector
    function move_to_position(destination)
    {
        this.bot.SetLocalOrigin(destination)
    }


    // Moves the bot to a random location within a sphere of radius RADIUS around ORIGIN
    function move_to_random_position()
    {
        local radius = SPAWN_RADIUS;
        local origin = Vector(ORIGIN_X, ORIGIN_Y, STATUE_HEIGHT + SPAWN_RADIUS);

        local dir = Vector();
        VS.RandomVectorInUnitSphere(dir);
        dir *= radius;

        local targetPos = origin + dir;

        move_to_position(targetPos);
    }
}

// returns string containing components of vec
function vector_to_string(vec)
{
    return format("%.3f %.3f %.3f", vec.x, vec.y, vec.z);
}

// ------------------------Target bot creation-------------------------

::spawnedBot <- TargetBot()

// --------------------------Event collection---------------------------

local EventsID = UniqueString()
getroottable()[EventsID] <-
{

    OnScriptHook_OnTakeDamage = function(params) {
        // this takes the longest

        // checks if damage was done to target_bot
        if (params.const_entity == spawnedBot.bot) {
            if (debug) printl("target_bot took damage")

            local attacker = params.attacker
            local damage = params.damage + params.damage_bonus

            if (attacker.IsPlayer() && attacker != null)
            {
                if (debug) printl(attacker + " did " + damage + " dmg" + "\nBots health: " + spawnedBot.health)

                // append_to_file("squirrel_out",  format("d %d %f", attacker.entindex(), damage))
                spawnedBot.damage_register +=  format("d %d %f\n", attacker.entindex(), damage)

                if(spawnedBot.auto_reposition)
                {
                    // reposition logic
                    spawnedBot.health -= damage
                    if (MAX_HEALTH - spawnedBot.health >= spawnedBot.reposition_damage)
                    {
                        if (debug) printl("reposition")
                        spawnedBot.move_to_random_position()
                        spawnedBot.health = MAX_HEALTH
                    }
                }
            }
        }
        else
        {
            if (debug) printl("damage to other entity")
        }
        params.damage = 0
    }

    // SendPositions hook
    OnScriptHook_SendPositions = function(_) {
        // sending target_bot position to squirrel_out
        local complete_positions_data_string = ""
        // appending target bot position
        complete_positions_data_string += "p t " + spawnedBot.bot.entindex() + " " + vector_to_string(spawnedBot.bot.GetOrigin()) + "\n"
        // appending all shooter bots positions
        complete_positions_data_string += shooter_bots_positions_to_string("p s")
        // appending to the out file
        append_to_file("squirrel_out", complete_positions_data_string)

        if (debug) printl(complete_positions_data_string)
    }

    // Reposition hook
    OnScriptHook_Reposition = function(_) {
        spawnedBot.move_to_random_position()
    }

	// Disabling hooks
	OnScriptHook_KillTargetBot = function(_) {
        printl("Deleting hooks for targetBot")
        delete getroottable()[EventsID]
    }

    // SendDamage hook
    OnScriptHook_SendDamage = function(_) {
    if(debug) printl("SendDamage hook")

	if(spawnedBot.damage_register == "") 
    {
		// no damage case
		append_to_file("squirrel_out", "d none")
	}
	else 
    {
		// damage done
		append_to_file("squirrel_out", spawnedBot.damage_register)
	}

        spawnedBot.damage_register = ""
    }
}

local EventsTable = getroottable()[EventsID]
foreach (name, callback in EventsTable) EventsTable[name] = callback.bindenv(this)
__CollectGameEventCallbacks(EventsTable)
