::FLT_MAX <- 3.402823466e+38
::TF_TEAM_RED <- 2
::GRAV_MIN <- 0.00001

// Debugging constant
const debug = true

// Bot spawn constants
const DMG_BEFORE_REPOSITION = 100 // TODO specify how much damage for bot to reposition

const SPAWN_RADIUS = 100
const ORIGIN_X = 0
const ORIGIN_Y = 0

class ::TargetBot
{
    bot = null
    bot_pos = null
    seq_idle = null
    health = 10000
    reposition_damage = DMG_BEFORE_REPOSITION
    max_health = 10000

    // nested contructor handling 2 cases:
    // 1) the bot already exists on server -> getting the bot entity
    // 2) the bot doesn't exist -> spawning it from scratch
    constructor(arg)
    {
        if (typeof arg == "instance" && arg.IsValid())
        {
            // existing bot
            this.bot = arg
        }
        else if (typeof arg == "Vector")
        {
            // new spawn
         this.bot = SpawnEntityFromTable("base_boss",
            {
                targetname = "target_bot",
                origin = arg,
                model = "models/bots/soldier/bot_soldier.mdl",
                playbackrate = 1.0,
                health = 10000
            })
        }

        this.bot.AcceptInput("SetStepHeight", "18", null, null)
        this.bot.ValidateScriptScope()
        local scope = this.bot.GetScriptScope()
        scope.bot_brain <- this

        this.bot_pos = this.bot.GetOrigin()
        this.seq_idle = this.bot.LookupSequence("Stand_MELEE")

        printl("Grav contrustor" + this.bot.GetGravity())
    }

    // moves bot to destination Vector and logs position to a file
    function move_to_position(destination)
    {
        this.bot.SetLocalOrigin(destination)
    }


    // Moves the bot to a random location within a sphere of radius RADIUS around ORIGIN
    function move_to_random_position()
    {
        local radius = SPAWN_RADIUS // you can tweak this
        local origin = Vector(ORIGIN_X, ORIGIN_Y, 2*SPAWN_RADIUS)
        // Generate random point in unit sphere
        local dir;
        do {
            dir = Vector(
                RandomFloat(-1.0, 1.0),
                RandomFloat(-1.0, 1.0),
                RandomFloat(-1.0, 1.0)
            );
        } while (dir.LengthSqr() > 1.0); // reject points outside the unit sphere

        // Scale to desired radius
        dir *= RandomFloat(0.0, radius);

        // Final position
        local targetPos = origin + dir;

        move_to_position(targetPos);
        printl("Grav move to random" + this.bot.GetGravity())
    }

}

// --------------------------Helper functions--------------------------

// gets random value in a specified range
function get_random_in_range(min, max)
{
    return min + rand() % (max - min + 1);
}

// returns instance of target_bot on the server
function get_instance()
{
    local bot = Entities.FindByName(null, "target_bot");
    if (bot == null) {
        if (debug) printl("spawning target_bot")
        bot = TargetBot(Vector(0, 0, 64));
        return bot
    }
    else {
        if (debug) printl("recreating target_bot")
        local recreated_bot = TargetBot(bot)
        return recreated_bot
    }
}

// returns string containing components of vec
function vector_to_string(vec)
{
    return format("%.3f %.3f %.3f", vec.x, vec.y, vec.z);
}

// ------------------------File helper functions------------------------

// returns content of a file as a string
function read_from_file(filename)
{
    local string = null;
    try {
        string = FileToString(filename);
    } catch (e) {
        printl("Error reading from file: " + filename);
    }
    return string;
}

// sends text to file specified in filename
// file gets created in tf/scriptdata folder
function send_to_file(filename, string)
{
    if (typeof string == "string")
        StringToFile(filename, string)
    else
        printl("send_to_file: not a string")
}

// appends text to file specified in filename
function append_to_file(filename,  textToAppend)
{
    if (debug) printl("trying to append: " + textToAppend)
    if (typeof textToAppend != "string") "append: not a string"

    local file_contents = read_from_file(filename)
    if (file_contents == null)
        file_contents = textToAppend
    else {
        file_contents += ("\n" + textToAppend)
    }
    send_to_file(filename, file_contents)
}

// -------------------------------Calls--------------------------------

::spawnedBot <- get_instance()
spawnedBot.bot.SetGravity(0.00001)
if(!debug)
    spawnedBot.move_to_random_position()
else
{
    spawnedBot.move_to_random_position()
}

// --------------------------Event collection---------------------------

// registers game events
function collect_events_in_scope(events)
{
    local events_id = UniqueString();
    local events_table = {};

    // Bind all event callbacks to the current object context
    foreach (name, callback in events)
        events_table[name] <- callback.bindenv(getroottable());

    // Register the events
    __CollectGameEventCallbacks(events_table);

    // Optionally store the table in root if you want external reference
    getroottable()[events_id] <- events_table;
}

function shooter_bots_positions_to_string(prefix)
{
    // get all shooter bots
    local bot_list = []
    local ent = null
    while (ent = Entities.FindByClassname(ent, "player")) { //our bots are player class
        //printl(ent)
        // lets assume that bots are in RED team
        if (ent.GetTeam() == TF_TEAM_RED) {
            bot_list.append(ent)
        }
    }
    // get their postitions and put them to string
    local return_string = ""
    foreach(i, ent in bot_list)
    {
        return_string += prefix + " " + ent.entindex() + " " + vector_to_string(ent.GetOrigin()) + "\n"
    }
    // return the string
    return return_string
}


// collecting events
collect_events_in_scope({
    //called when target_bot gets damaged
    function OnScriptHook_OnTakeDamage(params)
	{
        // checks if damage was done to target_bot
        if (params.const_entity == spawnedBot.bot) {
            if (debug) printl("target_bot took damage")

            local attacker = params.attacker
            local damage = params.damage + params.damage_bonus

            if (attacker.IsPlayer() && attacker != null)
            {
                if (debug) printl(attacker + " did " + damage + " dmg")
                if(debug) printl("Bots health: " + spawnedBot.health)

                append_to_file("squirrel_out",  format("d %d %f", attacker.entindex(), damage))

                // reposition logic
                spawnedBot.health -= damage
                if (spawnedBot.max_health - spawnedBot.health >= spawnedBot.reposition_damage)
                {
                    if (debug) printl("reposition")
                    spawnedBot.move_to_random_position()
                    spawnedBot.health = spawnedBot.max_health
                }
            }
        }
        else
        {
            if (debug) printl("damage to other entity")
        }
	}

    // called when a script requests for postition_data
    function OnScriptHook_SendPositions(params) {
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

    function OnScriptHook_Reposition(params) {
        spawnedBot.move_to_random_position()
    }
});
