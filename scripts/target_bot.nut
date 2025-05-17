::FLT_MAX <- 3.402823466e+38

// Debugging constant
const debug = false

// Bot spawn constants
const X_MIN = 100 // TODO specify where target_bot can get spawned
const X_MAX = 300
const Y_MIN = 100
const Y_MAX = 300
const Z_MIN = 0.001
const Z_MAX = 5
const DMG_BEFORE_REPOSITION = 100 // TODO specify how much damage for bot to reposition

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
                health = FLT_MAX
            })
        }

        this.bot.AcceptInput("SetStepHeight", "18", null, null)
        this.bot.ValidateScriptScope()
        local scope = this.bot.GetScriptScope()
        scope.bot_brain <- this

        this.bot_pos = this.bot.GetOrigin()
        this.seq_idle = this.bot.LookupSequence("Stand_MELEE")
    }

    // moves bot to destination Vector and logs position to a file
    function move_to_position(destination)
    {
        this.bot.SetLocalOrigin(destination)
        append_to_file("target_bot_position_data", format("%s", vector_to_string(destination)))
    }

    // moves the bot to a random location
    function move_to_random_position()
    {
        local x = get_random_in_range(X_MIN, X_MAX)
        local y = get_random_in_range(Y_MIN, Y_MAX)
        local z = get_random_in_range(Z_MIN, Z_MAX)
        move_to_position(Vector(x,y,z))
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
if(!debug)
    spawnedBot.move_to_random_position()
else
{
    // test purposes
    spawnedBot.move_to_position(Vector(273, 503, 0.03125))
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

                append_to_file("target_bot_dmg_data",  format("%d %f", attacker.entindex(), damage))

                // reposition logic
                spawnedBot.health -= damage
                if (spawnedBot.max_health - spawnedBot.health >= spawnedBot.reposition_damage)
                {
                    if (debug) printl("reoposition")
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

    // function OnGameEvent_npc_hurt(params)
	// {

    //     local attacker = EntIndexToHScript(params.attacker_player)
    //     local damage = params.damageamount

    //     printl("on  npc hurt")
    //     if (attacker.IsPlayer() && attacker != null)
	// 	{
	// 		printl(get_id(attacker) + " did " + damage + " dmg")
    //         append_to_file("target_bot_data",  format("%d %f", get_id(attacker), damage))
	// 	}
	// }
});