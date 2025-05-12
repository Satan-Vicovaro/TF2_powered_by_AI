
function LogPlayerPosition() {
    local player = Entities.FindByClassname(null, "player");
    if (player != null) {
        local pos = player.GetOrigin();
        AddThinkToEnt(player, "PrintPlayerPosition")
        local msg = format("PlayerPos: x=%.2f y=%.2f z=%.2f", pos.x, pos.y, pos.z);
        printl(msg); // prints to console/log
    }
}
print("lol")
LogPlayerPosition()