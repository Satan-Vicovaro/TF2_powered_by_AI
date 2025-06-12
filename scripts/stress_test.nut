::FIRE_RATE <- 0.05; // seconds between shots per bot (adjust as needed)
::fire_logic <- null;
::iterations <- 0;
const debug_fire = true;


function start_firing_loop()
{
    // Clean up any existing fire_logic
    if ("fire_logic" in getroottable() && ::fire_logic != null && ::fire_logic.IsValid())
    {
        NetProps.SetPropString(::fire_logic, "m_iszScriptThinkFunction", "");
        ::fire_logic.Destroy();
        ::fire_logic <- null;
    }

    // Create logic_script for firing loop
    ::fire_logic = Entities.CreateByClassname("logic_script");
    ::fire_logic.ValidateScriptScope();
    ::fire_logic.KeyValueFromString("targetname", "bot_fire_logic");

    local scope = ::fire_logic.GetScriptScope();
    scope["BotFireThink"] <- BotFireThink;

    AddThinkToEnt(::fire_logic, "BotFireThink");

    printl("Started bot firing loop.");
}

function stop_firing_loop()
{
    if (::fire_logic != null && ::fire_logic.IsValid())
    {
        NetProps.SetPropString(::fire_logic, "m_iszScriptThinkFunction", "");
        ::fire_logic.Destroy();
        ::fire_logic <- null;

        printl("Bot firing loop stopped.");
    }
    ::iterations <- 0;
}

function BotFireThink()
{
    local ent = null;
    local count = 0;
    ::iterations++;
    while (ent = Entities.FindByClassname(ent, "player"))
    {
        if (ent == GetListenServerHost()) continue;
        if (!ent.IsValid()) continue;
        if (!ent.IsAlive()) continue;

        local wep = ent.GetActiveWeapon();
        if (wep && wep.IsValid())
        {
            wep.PrimaryAttack();
            wep.SetClip1(wep.GetMaxClip1())
            count++;
        }
    }

    if(debug_fire) printl("Iterations: " + ::iterations);
    return ::FIRE_RATE;
}