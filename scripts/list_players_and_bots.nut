::TF_TEAM_RED <- 2

function print_current_players()
{
    printl("Player list:")
    local maxPlayers = MaxClients().tointeger();

    for (local i = 1; i <= maxPlayers; i++)
    {
        local player = PlayerInstanceFromIndex(i);
        if (player != null && player.IsValid())
        {
            local pos = player.GetOrigin();
            printl("Player " + i + " entindex: " + player.entindex() + " position: " + pos.tostring());
        }
    }
    printl("")
}

function get_bots()
{
    local bot_list = []
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

function print_bots()
{
    printl("Bot list:")
    local bot_list = get_bots()
    foreach(i, ent in bot_list)
    {
        local pos = ent.GetOrigin();
        printl("bot " + i + " " + ent.entindex() + " " + pos)
    }
    printl("")
}

print_current_players()
print_bots()